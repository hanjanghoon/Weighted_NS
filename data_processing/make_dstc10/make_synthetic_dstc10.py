import json
from numpy.core.arrayprint import set_string_function
from regex import D
from tqdm import tqdm
import re
import numpy as np
import random
import copy
from sentence_transformers import SentenceTransformer, util
import torch
import os
random.seed(1228)
np.random.seed(1228)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="8"
version="weighted"
domain_text={
            "hotel":["hotel","guesthouses","guest house","guesthouse"],
            "restaurant":["food","restaurant","eat"]
            }
kb9_entity_variation={
    "hotel" : ['alexander','arbury','aylesbray','carolina','city centre north','express by holiday inn',
    'finches','hamilton','home from home','lovell','the cambridge belfry','cambridge belfry','a and b ', 'acorn ', 'alpha-milton',
    'arbury lodge guest', 'archway ', 'ashley ', 'autumn ', 'aylesbray lodge ', 'bridge ', 'gonville ', 
    'hobsons ', 'huntingdon marriott ', 'kirkwood ', 'leverton ', 'lime', 'the lensfield ', 'university arms ', 'warkworth ', 'worth'],
    "restaurant" : ['midsummer','alimentum','bloomsbury','shanghai family','shanghai','efes','meze','de luca cucina','chiquito restaurant',
    'chiquito','frankie and bennys','pizza hut cherry hinton','lucky star','gourmet burger kitchen','pizza hut city centre','one seven',
    'little seoul','shiraz','gandhi','sesame restaurant','sesame','fitzbillies','dojo','michaelhouse','ugly duckling','varsity',
    'stazione restaurant','stazione','jinling','peking','maharajah tandoori','cambridge lodge','hotpot','city stop','nirala','two two',
    'grafton','pipasha','zizzi','missing sock','cow pizza kitchen']
}
kb10_entity_variation={
    "hotel" : ['a la turca'],
    "restaurant" : ['midsummer','alimentum','bloomsbury','shanghai family','shanghai','efes','meze','de luca cucina','chiquito restaurant',
    'chiquito','frankie and bennys','pizza hut cherry hinton','lucky star','gourmet burger kitchen','pizza hut city centre','one seven',
    'little seoul','shiraz','gandhi','sesame restaurant','sesame','fitzbillies','dojo','michaelhouse','ugly duckling','varsity',
    'stazione restaurant','stazione','jinling','peking','maharajah tandoori','cambridge lodge','hotpot','city stop','nirala','two two',
    'grafton','pipasha','zizzi','missing sock','cow pizza kitchen']
}
kb9_entity_variation['hotel']=sorted(kb9_entity_variation['hotel'],key=lambda x:len(x),reverse=True)
kb9_entity_variation['restaurant']=sorted(kb9_entity_variation['restaurant'],key=lambda x:len(x),reverse=True)



class Data_construction():
    def __init__(self):
        self.file_open()
        
        
        self.kb9_entity_name=self.get_kb9_all_entity_name()
        domain_session_logs=self.extract_domain_session()
        self.save_domain_session_logs(domain_session_logs)

        self.seen_entity_num=3
        
        self.dstc9_lastur=self.make_dstc9_lastur()
        self.kb9_unique_faq,self.kb10_unique_faq,self.kb10_all_entities=self.find_unique_faq()
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
        kb10_mapping=self.mapping_k10tok9()
        self.kb10_mapping=self.mapping_k10tok10(kb10_mapping)
        self.kb10_map_uttr=self.mapping_kb10_utterance()

        self.entity_exist=0
        self.entity_none=0

        augmented_domain_session_logs=self.insert_augmented_ur()
        self.split_dialog_session(augmented_domain_session_logs)

        self.add_other_data()

    def file_open(self):
        with open('make_dstc10/dstc10_knowledge_original.json', 'r') as f:
            self.kb10 = json.load(f)

        with open('dstc9_data/test/knowledge.json', 'r') as f:
            self.kb9 = json.load(f)

        #전체 대화는 71348
        with open('dstc9_data/train/logs.json', 'r') as f:
            train_logs9 = json.load(f)

        #label은 외부지식 사용할때만....
        with open('dstc9_data/train/labels.json', 'r') as f:
            train_labels9 = json.load(f)

        with open('dstc9_data/val/logs.json', 'r') as f:
            val_logs9 = json.load(f)

        #label은 외부지식 사용할때만....
        with open('dstc9_data/val/labels.json', 'r') as f:
            val_labels9 = json.load(f)

        with open('dstc9_data/test/logs.json', 'r') as f:
            test_logs9 = json.load(f)

        #label은 외부지식 사용할때만....
        with open('dstc9_data/test/labels.json', 'r') as f:
            test_labels9 = json.load(f)

        self.logs9= train_logs9+val_logs9+test_logs9
        self.labels9= train_labels9+val_labels9+test_labels9
    
    def get_kb9_all_entity_name(self):
        kb9_domain_entity_dict={}
        for domain in self.kb9:
            kb9_domain_entity_dict[domain]=[]
            for entity in self.kb9[domain]:
                if entity=='*':#train taxi
                    continue
                else:
                    # kb9list.append(kb9[domain][entity]['name'].lower())
                    kb9_domain_entity_dict[domain].append(self.kb9[domain][entity]['name'].lower())
        return kb9_domain_entity_dict

    def check_entity_in_uttr(self,uttr_dict):
        self.find_domain=False
        for kb_domain in self.kb9_entity_name:
            for entity in self.kb9_entity_name[kb_domain]:#식당 호텔 기차 택시
                if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                    uttr_dict["domain"]=kb_domain
                    #entity는 __domain__으로 바꿔줌.
                    uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                    self.domain_session.append(uttr_dict)
                    self.previous_domain=kb_domain
                    self.find_domain=True
                    self.find_entity=True
                    return
                    
            
            #만약 생략된거나 변형된거라면? 변형은  b & b나 -이런것
            if kb_domain in kb9_entity_variation:
                for entity in kb9_entity_variation[kb_domain]:
                    entity=entity.strip()
                    if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                        if re.search(("\S"+entity),uttr_dict['text'].lower()) is not None:
                            continue
                        uttr_dict["domain"]=kb_domain
                        #entity는 __domain__으로 바꿔줌.
                        uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                        self.domain_session.append(uttr_dict)
                        self.previous_domain=kb_domain
                        self.find_domain=True
                        self.find_entity=True
                        return

    def check_keyword_in_uttr(self,uttr_dict):
         #텍스트에서 키워드를 통해 스위칭 되는걸 찾아보자 스위칭 될경우 도메인을 유지하면 안된다.
        diff_domain_flag=False
        for txt_domain in domain_text:
            for keyword in domain_text[txt_domain]:
                if keyword in uttr_dict['text'].lower():#before entity가 false는 entity가 등장하지 않았으면 도메인이 확실하여도 치환할수 없다.
                    if txt_domain!=self.previous_domain or self.find_entity==False:#이 경우 domain switching을 의미하지만 entity가 없으므로 특정 도메인이라고 해주긴 어렵다.
                        # uttr_dict["domain"]="uncertain"+"_"+txt_domain
                        uttr_dict["domain"]=txt_domain
                        diff_domain_flag=True
                    else:
                        uttr_dict["domain"]=self.previous_domain
                        self.domain_session.append(uttr_dict)
                        return 
        
        #다 돌아도 모두다 다른 키워드일경우
        if diff_domain_flag==True:    
            self.domain_session.append(uttr_dict)
            self.previous_domain=uttr_dict["domain"]
            self.find_entity==False


        else:
        #나머지의 경우 이전 domain을 따라감
            uttr_dict["domain"]=self.previous_domain
            self.domain_session.append(uttr_dict)            

    def extract_domain_session(self):
        target_list={}
        previous=None
        dialog_cnt=0
        domain_session_logs=[]
        for dialog,dialabel in tqdm(zip(self.logs9,self.labels9),total=len(self.logs9)):

            #다음 대화 세션으로 넘어왓다는뜻.
            if previous:
                #가장 긴 대화 하나에 대해서만 만들거임.
                if previous[0]['text']!=dialog[0]['text']:
                    dialog_cnt+=1
                    self.domain_session=[]
                    self.previous_domain="uncertain"
                    self.find_entity=False
                    for i,uttr_dict in enumerate(previous):
                        
                        # remove origianal ur need kb10
                        if i in target_list:
                            uttr_dict["domain"]=target_list[i]['domain']
                            self.find_entity=True
                        
                        # remove origianal ur need kb10
                        elif i-1 in target_list:
                            uttr_dict["domain"]=target_list[i-1]['domain']
                            self.find_entity=True    
                        else:
                            self.find_domain=False
                            #check entity name
                            self.check_entity_in_uttr(uttr_dict)
                            #check keyword
                            if self.find_domain == False:
                                self.check_keyword_in_uttr(uttr_dict)

                    domain_session_logs.append(self.domain_session)
                    target_list={}

            previous=dialog
            #만약 도메인이 있다면. user-trun임.
            if dialabel['target']:
                for k_dict in dialabel['knowledge']:
                    target_list[len(dialog)-1]=k_dict

        #for last session.
        dialog_cnt+=1
        self.domain_session=[]
        self.previous_domain="uncertain"
        self.find_entity=False
        for i,uttr_dict in enumerate(previous):
            
            # remove origianal ur need kb10
            if i in target_list:
                uttr_dict["domain"]=target_list[i]['domain']
                self.find_entity=True
            
            # remove origianal ur need kb10
            elif i-1 in target_list:
                uttr_dict["domain"]=target_list[i-1]['domain']
                self.find_entity=True    
            else:
                self.find_domain=False
                #check entity name
                self.check_entity_in_uttr(uttr_dict)
                #check keyword
                if self.find_domain == False:
                    self.check_keyword_in_uttr(uttr_dict)

        domain_session_logs.append(self.domain_session)
        target_list={}
        return domain_session_logs

    def save_domain_session_logs(self,domain_session_logs):
        with open('make_dstc10/multiwoz_attraction_only_dialogue.json', 'r') as f:
            attract_single = json.load(f)

        with open('make_dstc10/multiwoz_attraction_cross_dialogue.json', 'r') as f:
            attract_cross = json.load(f)
                
        domain_session_all=[]
        print("cross: %d, single:%d origin: %d"%(len(attract_cross),len(attract_single),len(domain_session_logs)))

        for dialog in domain_session_logs+attract_single+attract_cross:
            domain_session_all.append(dialog)

        try:
            if not os.path.exists("make_dstc10/mydata/%s"%version):
                os.makedirs("make_dstc10/mydata/%s"%version)
        except OSError:
            print ('Error: Creating directory. ' +  "make_dstc10/mydata/newlog%d"%version)

        with open('make_dstc10/mydata/%s/domain_session_all.json'%version, 'w') as f:
            json.dump(domain_session_all, f, indent=4)


    def replace_entity(self,text,entity_name,domain):
        
        text=copy.deepcopy(text.lower())
        entity_name=entity_name.lower()
        if entity_name in text:
            text=text.replace(entity_name,"__%s__"%domain)
        else:
            if domain!='attraction':
                for vari_enitity in kb9_entity_variation[domain]:
                    if vari_enitity in text:
                        text=text.replace(vari_enitity,"__%s__"%domain)
                        break
        text=' '.join(text.split())#공백 제거
        text=text.replace(" ?","?")#공백 제거
        text=text.strip()
    
        return text
    
    def make_dstc9_lastur(self):
        # os.makedirs("make_dstc10/mydata/%s"%version,exist_ok=True)
        valid_turn_cnt=0
        dstc9_lastur={} #{'title':{},'body':{}}
        #정답 매칭 
        for log,label in zip(self.logs9,self.labels9):
            if label['target']==True:
                
                if len(label['knowledge'])>1:
                    print("wowowowwow")
                
                domain=label['knowledge'][0]['domain']
                entity_id=label['knowledge'][0]['entity_id']
                doc_id=label['knowledge'][0]['doc_id']
                

                if entity_id=='*':#taxi,train 일단 skip
                    continue
                valid_turn_cnt+=1
                
                key="%s__%s__%s"%(domain,entity_id,doc_id)

                if key not in dstc9_lastur:
                    dstc9_lastur[key]={"title":[],"body":[] }

                try:
                    user_text=log[-1]['text'].lower()
                except:
                    user_text=log[0]['text'].lower()
                
                system_text=label['response'].lower()

                entity_name=self.kb10[domain][str(entity_id)]['name'].lower()

                user_text=self.replace_entity(user_text,entity_name,domain)
                system_text=self.replace_entity(system_text,entity_name,domain)
                
                #문장 추출. faq에 대한 dstc9의 문장을 맵핑함. 이때 엔티티 정보는 제거하고 맵핑함.그래야 다른 엔티티를 끼울수 있다.
                dstc9_lastur[key]['title'].append(user_text)
                dstc9_lastur[key]['body'].append(system_text)

        return dstc9_lastur
    ###########################################################################
    ##1. dstc9 대화로 부터 faq에 대응되는 발화 추출하기 이때 entitiy는 제거해서##
    ##########################################################################

    #dstc9 대화 데이터로 부터 사용할 문장을 추출할대 쓰임.
    def insert_faq(self,domain,key,dl_faq,unqiue_faq):
        
        for type in ['title','body']:
            sentence=dl_faq[type]
            if sentence not in unqiue_faq[domain][type]:
                unqiue_faq[domain][type][sentence]=[]
            
            unqiue_faq[domain][type][sentence].append(key)
        
        return unqiue_faq


    def find_unique_faq(self):
        self.kb10_names=[]
        #dstc9 entity를 domain 별로 나눠보자
        kb9_unique_faq={}
        kb10_unique_faq={}
    
        kb10_all_entities=[]

        for domain in self.kb10:
            if domain=='taxi' or domain=='train':
                continue
            kb9_cnt=0
            kb10_cnt=0
            kb9_unique_faq[domain]={"title":{},"body":{}}
            kb10_unique_faq[domain]={"title":{},"body":{}}


            for entity_id in self.kb10[domain]:
                
                entity_name=self.kb10[domain][entity_id]['name'].lower()
                self.kb10_names.append(entity_name)
                #도메인별 엔티티 채워줌
                #kb10에서 나타난 모든 발화를 저장함
                #일단 질문으로만 가보자.
                for doc_id in self.kb10[domain][entity_id]['docs']:
                    title=self.kb10[domain][entity_id]['docs'][doc_id]['title'].lower()
                    body=self.kb10[domain][entity_id]['docs'][doc_id]['body'].lower()
                    
                    title=self.replace_entity(title,entity_name,domain)
                    body=self.replace_entity(body,entity_name,domain)
                    
                    #delexicaliezed faqs
                    dl_faqs={'title':title,'body':body}
                    key="%s__%s__%s"%(domain,entity_id,doc_id)

                    if self.kb10[domain][entity_id]['city']=="San Francisco" :
                       
                        kb10_all_entities.append(key)
                        kb10_cnt+=1

                        kb10_unique_faq=self.insert_faq(domain,key,dl_faqs,kb10_unique_faq)
                       
                       
                    #because of dstc9_test set, include all knowledge in kb9
                    kb9_cnt+=1
                    kb9_unique_faq=self.insert_faq(domain,key,dl_faqs,kb9_unique_faq)
                            
                        
                    #print("None")
            print("kb9 domain=%s unique title=%d unique body=%d all query=%d"%(domain,len(kb9_unique_faq[domain]['title']),len(kb9_unique_faq[domain]['body']),kb9_cnt))
            print("kb10 domain=%s unique title=%d unique body=%d all query=%d"%(domain,len(kb10_unique_faq[domain]['title']),len(kb10_unique_faq[domain]['body']),kb10_cnt))
            print()
            #hotel dialog = 3300, restaurnat 3300, attraction 1100
            #dialog 당 가용한 faq hotel = 8000, res=8000 attraction:2500
            #faq수  hotel=2000/3000 restauran=2800/5500 attraction=100/200
            #variation이 들어갈 경우..

            print("end mapping & query list")
        return kb9_unique_faq,kb10_unique_faq,kb10_all_entities



    def mapping_k10tok9(self):
        kb10_mapping={'title':{},'body':{}}
        num_k=10
        min_p=0.8
       
        for domain in self.kb10:
            # Two lists of sentences
            #debug
            if domain=='taxi' or domain=='train' :#or domain=='attraction':
                continue
            
            #type = title,body
            for type in kb10_mapping:
                kb9_faqs=list(self.kb9_unique_faq[domain][type].keys())
                #문제는 atrraction 어케할거냐?
                if not kb9_faqs:#attraction 일단 빠짐.
                    continue

                kb10_faqs=list(self.kb10_unique_faq[domain][type].keys())

                #Compute embedding for both lists
                kb9_faq_embeddings = self.model.encode(kb9_faqs, convert_to_tensor=True)
                kb10_faq_embeddings = self.model.encode(kb10_faqs, convert_to_tensor=True)

                cosine_scores_transpose = util.pytorch_cos_sim(kb10_faq_embeddings, kb9_faq_embeddings)
                topk_values_2d,topk_indices_2d=torch.topk(cosine_scores_transpose, k=num_k, dim=-1)

                for i,(topk_values,topk_indices) in enumerate(zip(topk_values_2d,topk_indices_2d)):
                    key10=kb10_faqs[i]
                    for topk_val, topk_idx in zip(topk_values,topk_indices):
                        if topk_val > min_p:
                            key9=kb9_faqs[int(topk_idx)]
                            for k10_keys in self.kb10_unique_faq[domain][type][key10]:
                                if k10_keys not in kb10_mapping[type]:
                                    kb10_mapping[type][k10_keys]={"dstc9ur_idx":[],"kb9":[],"kb10":[]}
                                #어자피 이거 묶음이라 원소 하나만 있어도됨. 그래서 앞 원소로 파악
                                if self.kb9_unique_faq[domain][type][key9][0] in kb10_mapping[type][k10_keys]['dstc9ur_idx']:
                                    break
                                else:
                                    kb10_mapping[type][k10_keys]['dstc9ur_idx'].extend(self.kb9_unique_faq[domain][type][key9])
                                    #entitiy 없으면 바로 문장으로 넣어줄거임.
                                    if key9 not in kb10_mapping[type][k10_keys]["kb9"]:
                                        kb10_mapping[type][k10_keys]["kb9"].append(key9)   
                        else:
                            break
            
        return kb10_mapping

    def mapping_k10tok10(self,kb10_mapping):

        num_k=20
        min_p=0.7
        for domain in self.kb10:
            
            if domain=='taxi' or domain=='train':
                continue
            
            for type in kb10_mapping:
                kb10_faqs=list(self.kb10_unique_faq[domain][type].keys())

                if not kb10_faqs:
                    continue

                model = SentenceTransformer('paraphrase-mpnet-base-v2')
                kb10_faq_embeddings = model.encode(kb10_faqs, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(kb10_faq_embeddings,kb10_faq_embeddings)
                topk_values_2d,topk_indices_2d=torch.topk(cosine_scores, k=num_k, dim=-1)
                for i,(topk_values,topk_indices) in enumerate(zip(topk_values_2d,topk_indices_2d)):
                    key10=kb10_faqs[i]
                    for topk_val, topk_idx in zip(topk_values,topk_indices):#여기서 나중에 top k
                        #print("%s \t %.3f"%(kb10_faqs[topk_idx],float(topk_val)))
                        key10_smil=kb10_faqs[int(topk_idx)]
                        if topk_val > min_p:
                            for k10_keys in self.kb10_unique_faq[domain][type][key10]:

                                if k10_keys not in kb10_mapping[type]:
                                    kb10_mapping[type][k10_keys]={"dstc9ur_idx":[],"kb9":[],"kb10":[]}

                                if key10_smil not in kb10_mapping[type][k10_keys]["kb10"]:
                                    kb10_mapping[type][k10_keys]["kb10"].append(key10_smil)

            
        return kb10_mapping

    def mapping_kb10_utterance(self):
        min_count=10
        kb10_map_uttr={'title':{},'body':{}}
        
        for type in kb10_map_uttr:
            
            total_cnt=0
            attr_variation_cnt=0
            attr_cnt=0

            for key in self.kb10_all_entities:
                
                variation_cnt=0


                if key not in kb10_map_uttr[type]:
                    kb10_map_uttr[type][key]=[]

                #if dstc9 lastur mapping with dstc10 faq exist.
                for dstc9_ur_key in self.kb10_mapping[type][key]['dstc9ur_idx']:
                    if dstc9_ur_key in self.dstc9_lastur:
                        dstc9_log_list=self.dstc9_lastur[dstc9_ur_key][type]
                        kb10_map_uttr[type][key].extend(dstc9_log_list)
                        variation_cnt+=len(dstc9_log_list)
                
                #if dstc9 faq mapping with dstc10 faq exist
                if variation_cnt<min_count:
                    kb9_faq_list=self.kb10_mapping[type][key]['kb9']
                    kb10_map_uttr[type][key].extend(kb9_faq_list)
                    variation_cnt+=len(kb9_faq_list)
                    #print("\ndstc9_faq=%d"%variation_cnt)
                
                #if other dstc10 faq exist       
                if variation_cnt<min_count:
                    kb10_faq_list=self.kb10_mapping[type][key]['kb10']
                    kb10_map_uttr[type][key].extend(kb10_faq_list)
                    variation_cnt+=len(kb10_faq_list)
                
                #나머지
                if variation_cnt<min_count:
                    print("cautious pair (%s) = %d"%(key,variation_cnt))
                
                total_cnt+=variation_cnt

                #domain
                if key.split("__")[0]=='attraction':
                    attr_variation_cnt+=variation_cnt
                    attr_cnt+=1
                #print(cnt)
            #title, body averarge
            print("\n%s\nvariation average = %.2f\n"%(type,total_cnt/(len(self.kb10_all_entities))))
            print("attr variation average = %.2f\n"%(attr_variation_cnt/attr_cnt))
        
        return kb10_map_uttr

    def make_single_faq_dialog(self,dialog,faqs,entity_idx,doc_idx_list,selected_uttr_pos,domain,entity_name):
        for _ in range(20):
            dialog=copy.deepcopy(dialog)
            selected_uttr_pos=sorted(selected_uttr_pos)
            entity_name=entity_name.lower()
            offset = 0
            #faq 데이터 발화형식으로 변환 해주기
            uttr_faq_list=[]
            for doc_idx in doc_idx_list:
                user_faq_dict={}
                user_faq_dict['speaker']='U'
                #user_faq_dict['text']=faqs[str(i)]['title']
                key="%s__%s__%s"%(domain,entity_idx,doc_idx)
                user_faq_dict['text']=random.sample(self.kb10_map_uttr['title'][key],1)[0].lower()

                user_faq_dict['domain']=domain+'_faq'
                user_faq_dict['entity_id']=entity_idx
                user_faq_dict['doc_id']=str(doc_idx)

                sys_faq_dict={}
                sys_faq_dict['speaker']='S'

                if self.kb10_map_uttr['body'][key]:
                    sys_faq_dict['text']=random.sample(self.kb10_map_uttr['body'][key],1)[0].lower()
                else:
                    sys_faq_dict['text']=faqs[str(doc_idx)]['body'].lower()

                sys_faq_dict['domain']=domain+'_faq'
                sys_faq_dict['entity_id']=entity_idx
                sys_faq_dict['doc_id']=str(doc_idx)
                uttr_faq_list.append([user_faq_dict,sys_faq_dict])



            for i in range(len(selected_uttr_pos)):
                
                dialog.insert(selected_uttr_pos[i]+1+offset, uttr_faq_list[i][0])
                dialog.insert(selected_uttr_pos[i]+2+offset, uttr_faq_list[i][1])
                offset += 2
            #entity껴주기
            
            for i in range(1,len(dialog)):
                dialog[i]['text']=dialog[i]['text'].lower()
                dialog[i]['text']=dialog[i]['text'].replace("__"+domain+"__",entity_name)
                dialog[i]['text']=dialog[i]['text'].lower().replace('cambridge','san francisco')
            
            check_en=False
            for i in range(1,len(dialog)):
                if entity_name in dialog[i]['text'].lower():
                    check_en=True
            
            if check_en==False:
                print("single entity error")
                self.entity_none+=1
            else:
                self.entity_exist+=1
                break

        return dialog 


    def make_cross_faq_dialog(self,dialog,new_knowledge,cross_domain_dict):
        global entity_exist,entity_none
        flag=False
        for _ in range(20):
            dialog=copy.deepcopy(dialog)
            
            all_pos=[]
            for domain in cross_domain_dict:
                if cross_domain_dict[domain]:
                    all_pos+=list(cross_domain_dict[domain]['selected_pos'])
            all_pos=sorted(all_pos)
        

            offset = 0
            #faq 데이터 발화형식으로 변환 해주기
            cross_faq_dict={}

            for domain in cross_domain_dict:
                if cross_domain_dict[domain]:
                    cross_faq_dict[domain]={}

                    entity_idx=cross_domain_dict[domain]['entity_idx']
                    faqs=new_knowledge[domain][entity_idx]['docs']
                    entity_name=new_knowledge[domain][entity_idx]['name'].lower()
                    cross_faq_dict[domain]['entity_name']=entity_name

                    uttr_faq_list=[]
                    for doc_idx in cross_domain_dict[domain]['doc_idx_list']:
                        key="%s__%s__%s"%(domain,entity_idx,doc_idx)
                        user_faq_dict={}
                        user_faq_dict['speaker']='U'
                        #user_faq_dict['text']=hotel_faqs[str(i)]['title']
                        user_faq_dict['text']=random.sample(self.kb10_map_uttr['title'][key],1)[0].lower()
                        user_faq_dict['domain']=domain+'_faq'
                        user_faq_dict['entity_id']=entity_idx
                        user_faq_dict['doc_id']=str(doc_idx)

                        sys_faq_dict={}
                        sys_faq_dict['speaker']='S'
                        if self.kb10_map_uttr['body'][key]:
                            sys_faq_dict['text']=random.sample(self.kb10_map_uttr['body'][key],1)[0].lower()
                        else:
                            sys_faq_dict['text']=faqs[str(doc_idx)]['body'].lower()
                        sys_faq_dict['domain']=domain+'_faq'
                        sys_faq_dict['entity_id']=entity_idx
                        sys_faq_dict['doc_id']=str(doc_idx)
                        
                        uttr_faq_list.append([user_faq_dict,sys_faq_dict])
                    
                    cross_faq_dict[domain]['uttr_faq_list']=uttr_faq_list


            idx_dict={"hotel":0,"restaurant":0,"attraction":0}
            for i in range(len(all_pos)):
                for domain in cross_faq_dict:
                    if all_pos[i] in cross_domain_dict[domain]['selected_pos']:
                        dialog.insert(all_pos[i]+1+offset, cross_faq_dict[domain]['uttr_faq_list'][idx_dict[domain]][0])
                        dialog.insert(all_pos[i]+2+offset, cross_faq_dict[domain]['uttr_faq_list'][idx_dict[domain]][1])
                        idx_dict[domain]+=1
                        offset += 2
            
            #entity껴주기
            for i in range(1,len(dialog)):
                dialog[i]['text']=dialog[i]['text'].lower()

                for domain in cross_faq_dict:
                    dialog[i]['text']=dialog[i]['text'].replace("__%s__"%domain,cross_faq_dict[domain]['entity_name'])
                
                dialog[i]['text']=dialog[i]['text'].lower().replace('cambridge','san francisco')

            #entity 없나 혹시?
            for domain in cross_faq_dict:
                check_en=False
                for i in range(1,len(dialog)):
                    if cross_faq_dict[domain]['entity_name'] in dialog[i]['text'].lower():
                        check_en=True
                if check_en==False:
                    print("cross entity error")
                    self.entity_none+=1
                else:
                    self.entity_exist+=1
                    flag=True
            if flag==True:
                break

        return dialog 
    
    def insert_augmented_ur(self):

        with open('make_dstc10/mydata/%s/domain_session_all.json'%version, 'r') as f:
            self.domain_session_all = json.load(f)
        with open('make_dstc10/knowledge.json', 'r') as f:
            self.new_knowledge = json.load(f)
        
        augmented_domain_session_logs=[]
        entities_num={
            "hotel":len(self.new_knowledge['hotel']),
            "restaurant": len(self.new_knowledge['restaurant']),
            "attraction": len(self.new_knowledge['attraction'])
        }
        entities_list={
            "hotel":list(self.new_knowledge['hotel'].keys()),
            "restaurant": list(self.new_knowledge['restaurant'].keys()),
            "attraction": list(self.new_knowledge['attraction'].keys())
        }

        entity_list_idx={
            "hotel":0,
            "restaurant": 0,
            "attraction": 0
        }

        #매번 모든 엔티티를 볼경우 셔플링을 해준다.
        print("domain_session_all_length=%d\n"%len(self.domain_session_all))

        for dialog in tqdm(self.domain_session_all):
            possible_loc={'hotel':[],'restaurant':[],'attraction':[]}

            # cross 인가 single 인가?
            # 특정 엔티티가 직접적으로 언급이 되었는가?
            for i in range(1,len(dialog)-1):
                if dialog[i]['speaker']=='S':
                    if 'unc' in dialog[0]:
                        uttr_domain = dialog[i]['domain'].replace("uncertain_","")
                    else:
                        uttr_domain = dialog[i]['domain']
                    
                    if uttr_domain in possible_loc:
                        possible_loc[uttr_domain].append(i)

        
            cross_flag=False
            domain_cnt=0
            for domain in possible_loc:
                if possible_loc[domain]:
                    domain_cnt+=1
            
            if domain_cnt>1:
                cross_flag=True

            elif domain_cnt==0:
                # transport제거
                # augmented_domain_session_logs.append(dialog)
                continue
            

            if cross_flag:
                debug=0
                for _ in range(self.seen_entity_num):#식당이 더 많아서 이걸로 봄. 아니다 위에서 추출해서 맞춰놓음.
                    #entity의 faq 개수를 본다.
                    cross_domain_dict={'hotel':{},'restaurant':{},'attraction':{}}
                    for domain in possible_loc:
                        if possible_loc[domain] :
                            # while True:
                            entity_idx=entities_list[domain][entity_list_idx[domain]]
                            entity_list_idx[domain]+=1
                            if entity_list_idx[domain]>=entities_num[domain]:
                                random.shuffle(entities_list[domain])
                                entity_list_idx[domain]=0
                                # if entity_idx in self.kb10_map_uttr['title']:
                                #     break

                            faqs=self.new_knowledge[domain][entity_idx]['docs']
                            entity_name=self.new_knowledge[domain][entity_idx]['name']

                            # candidates_doc=list(self.kb10_map_uttr['title'][entity_idx].keys())
                            faq_len=len(faqs)

                            #섞으나 안섞으나 똑같음.
                            #순서대로 2개의 질문을 뽑고
                            
                            doc_idx_list=random.sample(range(faq_len),min(2,faq_len))
                            selected_pos=np.random.choice(possible_loc[domain],min(2,faq_len))
                            
                            cross_domain_dict[domain]['entity_idx']=entity_idx
                            cross_domain_dict[domain]['doc_idx_list']=doc_idx_list
                            cross_domain_dict[domain]['selected_pos']=selected_pos

                    faq_dialog=self.make_cross_faq_dialog(dialog,self.new_knowledge,cross_domain_dict)
                    augmented_domain_session_logs.append(faq_dialog)
            else:
                #넘으면 나중에 생각...
                debug=0
                for domain in possible_loc:
                    if possible_loc[domain]:
                        for _ in range(self.seen_entity_num):
                            #돌리면서 넘겨줌..
                            entity_idx=entities_list[domain][entity_list_idx[domain]]
                            entity_list_idx[domain]+=1
                            if entity_list_idx[domain]>=entities_num[domain]:
                                random.shuffle(entities_list[domain])
                                entity_list_idx[domain]=0
                                # if entity_idx in self.kb10_map_uttr['title']:
                                #     break
                            
                            faqs=self.new_knowledge[domain][entity_idx]['docs']
                            entity_name=self.new_knowledge[domain][entity_idx]['name']
                            faq_len=len(faqs)


                            #섞으나 안섞으나 똑같음.
                            #순서대로 2개의 질문을 뽑고
                            
                            doc_idx_list=random.sample(range(faq_len),min(3,faq_len))
                            selected_pos=np.random.choice(possible_loc[domain],min(3,faq_len))

                            #print(selected_uttr_pos)
                            #갯수중에 복원 추출로 랜덤하게 3개를 뽑는다.
                            faq_dialog=self.make_single_faq_dialog(dialog,faqs,entity_idx,doc_idx_list,selected_pos,domain,entity_name)
                            augmented_domain_session_logs.append(faq_dialog)
                
            
                #else : unc_transport
        print("augmented dialog 수 : %d"%(len(augmented_domain_session_logs)))
        return augmented_domain_session_logs
    
    
    
    def check_dialog_entity_name(self,dialog,name):
        last_k=-1
        for i,utter_dict in enumerate(dialog):
            text=utter_dict['text'].lower()
            if name.lower() in text:
                last_k=i
        if last_k!=-1:
            gap=len(dialog)-last_k
        else:
            gap=-1
        return gap


    def ret_all_knowledge_entity(self,knowledge):
        know_check_list=[]
        for domain in knowledge:
            for entity_id in knowledge[domain]:
                know_check_list.append(entity_id)

        return know_check_list 



    def random_sample(self,given_list,num):
        return_list=[]
        while True:
            picked_list=random.sample(given_list,min(len(given_list),num))
            num=num-len(picked_list)
            return_list.extend(picked_list)
            if num<=0:
                break
        return return_list
    
    
    def split_dialog_session(self,augmented_domain_session_logs):
        
        try:
            if not os.path.exists("make_dstc10/mydata/%s"%version):
                    os.makedirs("make_dstc10/mydata/%s"%version)
        except OSError:
            print ('Error: Creating directory. ' +  "mydata/newlog%d"%version)

        new_logs=[]
        new_labels=[]
        cnt=0

        all_knowledge_entity_id=self.ret_all_knowledge_entity(self.new_knowledge)

        for dialog in tqdm(augmented_domain_session_logs):
            dialog_form=[]
            faq_uttr_idx_list=[]
            normal_uttr_idx_list=[]
            neg_idx=0
            for i in range(1,len(dialog)):
                dialog_form.append({'speaker':dialog[i]['speaker'],'text':dialog[i]['text']})
                
                #user 발화 기준으로 자름
                if dialog[i]['speaker']=='U':
                    if 'entity_id' in dialog[i]:
                        #taxi, train이면 걍 negative sample로
                        if dialog[i]['entity_id']=='*':
                            normal_uttr_idx_list.append(i)
                        else:
                            faq_uttr_idx_list.append(i)
                    else:
                        normal_uttr_idx_list.append(i)

            for user_idx in faq_uttr_idx_list:
                # gap=-1
                # while gap!=-1:
                try:
                    start_idx=random.randrange(0,user_idx-1, 2)
                except:
                    start_idx=0

                target_domain= dialog[user_idx]['domain'].rstrip('_faq')
                target_entity_id=dialog[user_idx]['entity_id']
                target_doc_id=dialog[user_idx]['doc_id']
                target_name=self.new_knowledge[target_domain][str(target_entity_id)]['name']
                splited_dialog=dialog_form[start_idx:user_idx]
                
                if target_entity_id in all_knowledge_entity_id:
                    all_knowledge_entity_id.remove(target_entity_id)
                
                gap=self.check_dialog_entity_name(splited_dialog,target_name)
                if gap >=0:
                    new_logs.append(splited_dialog)
                    new_labels.append({
                    'target': True,
                    "knowledge": [
                        {
                        "domain": target_domain,
                        "entity_id": target_entity_id,
                        "doc_id": target_doc_id
                        }
                    ],
                    "response": dialog_form[user_idx]['text']
                    })
            
            if not faq_uttr_idx_list:
                #taxi 랑 train 그냥 스킵
                #print(dialog[0])
                cnt+=1
                continue
            # #negative sample for detection
            
            try:
                user_idx_list=self.random_sample(normal_uttr_idx_list,max(len(faq_uttr_idx_list),3))
            except:
                print('err')
            for user_idx in user_idx_list:
                #추가
                try:
                    start_idx=random.randrange(0,user_idx-1, 2)
                except:
                    start_idx=0

                new_logs.append(dialog_form[start_idx:user_idx])
                new_labels.append({'target': False})

        print("transport only=%d"%cnt)





        with open('make_dstc10/mydata/%s/logs.json'%version, 'w') as f:
            json.dump(new_logs, f, indent=4)
        print("# of new_logs %d"%(len(new_logs)))
        with open('make_dstc10/mydata/%s/labels.json'%version, 'w') as f:
            json.dump(new_labels, f, indent=4)
        print("# of new_labels %d"%(len(new_labels)))


        #print(dialog_cnt+1)
        #print(mixed_domain_cnt)
    def add_other_data(self):
        with open('make_dstc10/mydata/%s/logs.json'%version, 'r') as f:
            train10_logs = json.load(f)
        with open('make_dstc10/mydata/%s/labels.json'%version, 'r') as f:
            train10_labels = json.load(f)

        with open('data/other_data/dstc9_data/test/logs.json', 'r') as f:
            test9_logs = json.load(f)
        with open('data/other_data/dstc9_data/test/labels.json', 'r') as f:
            test9_labels = json.load(f)

        trim_test9_logs=[]
        trim_test9_labels=[]
        for log, label in zip(test9_logs,test9_labels):
            if 'sf' in label['source']:
                trim_test9_logs.append(log)
                trim_test9_labels.append(label)

        with open('dstc10_data_sample/val/logs.json', 'r') as f:
            val10_logs = json.load(f)
        with open('dstc10_data_sample/val/labels.json', 'r') as f:
            val10_labels = json.load(f)

        print("test9 length %d"%len(trim_test9_logs))

        new_logs=train10_logs+trim_test9_logs#+val10_logs
        new_labels=train10_labels+trim_test9_labels#+val10_labels

        with open('make_dstc10/mydata/%s/logs.json'%version, 'w') as f:
            json.dump(new_logs, f, indent=4)
        with open('make_dstc10/mydata/%s/labels.json'%version, 'w') as f:
            json.dump(new_labels, f, indent=4)

if __name__ == "__main__":
    data=Data_construction()
    print("END Program")

