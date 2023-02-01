import os
import json
import random
import logging
import sys

from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)
from transformers import RobertaTokenizer, BertTokenizer
from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]

#jang-r
def init_special_tokens_by_model(tokenizer):
    #if 'roberta' in model_name.lower() or 'bart' in model_name.lower():

    if issubclass(type(tokenizer), RobertaTokenizer) :
        SPECIAL_TOKENS['bos_token'] = '<s>'
        SPECIAL_TOKENS_VALUES[0] = '<s>'
        SPECIAL_TOKENS['eos_token'] = '</s>'
        SPECIAL_TOKENS_VALUES[1] = '</s>'
        SPECIAL_TOKENS['additional_special_tokens'][3] = '</s>'
        SPECIAL_TOKENS_VALUES[6] = '</s>'
  
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.knowledge, self.snippets = self._prepare_knowledge()#doc밖에 안들었음. snippet에..

        #jang
        with open('data/other_data/smil_knowledge.json', 'r') as f:
             self.smil_knowledge = json.load(f)

        self._create_examples()

    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            #jang 디버그용으로 짤라버림
            # if i>100:
            #    break
            
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = dict()
        for snippet in self.knowledge_docs:
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "", domain= "")#jang snippet["domain"] or
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        return knowledge, tokenized_snippets

    def _knowledge_to_string(self, doc, name="",domain=""):
        return doc["body"]

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task!="detection":
                continue
                # we only care about non-knowledge-seeking turns in turn detection task
                    
            

            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            #앞에서 부터 자르는 건데 거의 안자르는거임.
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 앞에서 부터 자른다 잘짜졌음.
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)
            #jang
            rel_knowledge=[]
            domain_candidates=[]
            knowledge_key=None
            if target:
            
                #jang
                post_flag=False
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    #jang
                    if self.args.task=="post-training" or self.args.task=="generation":
                        post_flag=True
                    else:
                        if not self.args.eval_all_snippets :
                            raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                        label["knowledge"] = [self.knowledge_docs[0]]
                    
                    
                #jang
                if post_flag==False:
                    knowledge = label["knowledge"][0]
                    knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])
                    # find snippets with same entity as candidates
                    prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
                    
                    
                    knowledge_candidates = [
                        cand
                        for cand in self.snippets.keys() 
                        if "__".join(cand.split("__")[:-1]) == prefix
                    ]
                    #jang 여기
                    #if self.split_type == "train" and self.args.negative_sample_method == "domain":
                    if self.args.task=="selection" and self.args.negative_sample_method == "weighted" and self.split_type=='train':
                        domain_candidates=[
                            cand
                            for cand in self.snippets.keys() 
                            if str(cand.split("__")[0])==knowledge["domain"]
                        ]
                        #jang
                        
                        rel_knowledge=self.smil_knowledge[knowledge_key]
                    
                    
                    
                    if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                        # if there's not enough candidates during training, we just skip this example
                        if len(knowledge_candidates) < self.args.n_candidates:
                            continue
                    used_knowledge = self.snippets[knowledge_key]
                    used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]#이거 body만
                #jang
                else:
                    knowledge_candidates = None
                    used_knowledge = [] 
            else:
                knowledge_candidates = None
                used_knowledge = []
            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "rel_knowledge":rel_knowledge,#jang
                "domain":domain_candidates,
                "knowledge_key":knowledge_key,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })
           

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        #jang
        sequence = [[self.bos]  + [self.knowledge_tag] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, token_type_ids, lm_labels


class ResponseGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}
        #uttr간 발화 구분이 없다.
        #마지막 발화 구분만 존재함 아님 스피커 구분 해줌 괜춘한듯. 화자 바꾸어도 될듯 마지막만 맞으면 됨
        sequence = [[self.bos]] + history[:-1] + [[self.knowledge_tag] + history[-1] + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker

        instance["input_ids"] = list(chain(*sequence))
        #이거는 좀 이상하긴 한데 0,1대신 speaker 정보를 추가함.
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).float()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info

#jang-r
class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        if self.args.negative_sample_method not in ["all", "weighted", "oracle"]:
            raise ValueError("negative_sample_method must be all, weighted, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name="", domain=""):
        join_str = " %s " % self.knowledge_sep_token
        if self.args.selection_type == "domain":
            return domain
        elif self.args.selection_type == "entity" or self.args.selection_type == "domain_entity":
            return join_str.join([domain, name])
        elif self.args.selection_type == "body":
            return join_str.join([name, doc["body"]])
        return join_str.join([domain, name, doc["title"], doc["body"]])

    def _split_int_array(self, seq, smallest):    
        group = []    
        for num in seq:
            if num != smallest:
                group.append(num)
            elif group:
                yield group
                group = []
        if group:
            yield group

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            if self.args.eval_all_snippets:
                candidate_keys = list(self.snippets.keys())
            else:
                #jang 여기
                random_cand=example["domain"]
                random_cand=random.sample(random_cand,k=len(random_cand)//3)
                
                if example["knowledge_key"] not in random_cand:
                    random_cand.append(example["knowledge_key"])
                
                candidate_keys=random_cand
                #candidate_keys = example["candidates"]+random.sample(list(self.snippets.keys()), \
                        #k=min(len(self.snippets.keys()),200))

            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]
        else:
            # if self.args.selection_type == "all":
            if self.args.negative_sample_method == "all":
                candidate_keys = list(self.snippets.keys())
                


            elif self.args.negative_sample_method == "weighted":
                # #jang 여기
                random_cand=random.sample(list(self.snippets.keys()), \
                    k=min(len(self.snippets.keys()),max(len(example["domain"]),self.args.n_candidates+1)))
                
                if example["knowledge_key"] in random_cand:
                    random_cand.remove(example["knowledge_key"])

                rel_cand=example["rel_knowledge"]
                if len(rel_cand)> 0:
                    rel_cand=rel_cand*(len(example["domain"])//len(rel_cand))

                if example["knowledge_key"] in example["domain"]:
                    example["domain"].remove(example["knowledge_key"])
                #jang
                candidate_keys = example["candidates"]*(len(example["domain"])//(2*len(example["candidates"]))) + random_cand+ rel_cand+ example["domain"] 
                
                
            
            elif self.args.negative_sample_method == "oracle":
                candidate_keys = example["candidates"]
            else: # although we have already checked for this, still adding this here to be sure
                raise ValueError("negative_sample_method must be all, weghted, or oracle, got %s" % self.args.negative_sample_method)
            
            
            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]
        

        this_inst["candidate_keys"] = candidate_keys

        if self.split_type == "train":
            # Sample args.n_candidates from candidates
            candidates = self._shrink_label_cands(example["knowledge"], candidates)

        label_idx = candidates.index(example["knowledge"])
            
        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
                #MMM [rm]
                # ,example["current_turn"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
        sequence = [[self.bos]] + history + [[self.knowledge_tag] + knowledge] + [[self.eos]]

        token_type_ids = [0 for _, s in enumerate(sequence) for _ in s]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = token_type_ids
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence
    
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        #MMM [md]
        token_type_pad = token_type_ids[0][-1] if self.args.type_vocab_size != self.args.vocab_size else self.pad
        #token_type_pad = self.pad
        
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, token_type_pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, lm_labels, label_idx, data_info