import json
from numpy.core.arrayprint import set_string_function
from tqdm import tqdm
import re
import numpy as np
import random
import copy
from sentence_transformers import SentenceTransformer, util
import torch
import os


#전체 대화는 71348
with open('data/train/logs.json', 'r') as f:
    train_logs9 = json.load(f)

#label은 외부지식 사용할때만....
with open('data/train/labels.json', 'r') as f:
    train_labels9 = json.load(f)

with open('data/val/logs.json', 'r') as f:
    val_logs9 = json.load(f)

#label은 외부지식 사용할때만....
with open('data/val/labels.json', 'r') as f:
    val_labels9 = json.load(f)

logs9=train_logs9 + val_logs9
labels9= train_labels9 +val_labels9




def find_entire_log():
    target_list={}
    previous=None
    dialog_cnt=0
    mixed_domain_cnt=0
    domain_session_logs=[]
    last_target=False
    for dialog,dialabel in tqdm(zip(logs9,labels9),total=len(logs9)):

        #다음 대화 세션으로 넘어왓다는뜻.
        if previous:
            if previous[0]['text']!=dialog[0]['text']:
                if last_target==True:
                    print('error')
                dialog_cnt+=1
                domain_session=[]
                previous_domain="uncertain"
                for i,uttr_dict in enumerate(previous):
                    uttr_dict['text']=uttr_dict['text'].lower()
                    #domain이 명시된 경우
                    if i in target_list:
                        uttr_dict["domain"]=target_list[i]['domain']
                        uttr_dict["entity_id"]=target_list[i]['entity_id']
                        uttr_dict["doc_id"]=target_list[i]['doc_id']
                        uttr_dict["type"]='title'

                    elif i-1 in target_list:
                        uttr_dict["domain"]=target_list[i-1]['domain']
                        uttr_dict["entity_id"]=target_list[i-1]['entity_id']
                        uttr_dict["doc_id"]=target_list[i-1]['doc_id']
                        uttr_dict["type"]='body'
                
                    domain_session.append(uttr_dict)
                
                target_list={}
                domain_session_logs.append(domain_session)
        previous=dialog
        
        if dialabel['target']:
            for k_dict in dialabel['knowledge']:
                target_list[len(dialog)-1]=k_dict
            last_target=True
        else:
            last_target=False

    domain_session=[]
    
    for i,uttr_dict in enumerate(previous):
        uttr_dict['text']=uttr_dict['text'].lower()
        #domain이 명시된 경우
        if i in target_list:
            uttr_dict["domain"]=target_list[i]['domain']
            uttr_dict["entity_id"]=target_list[i]['entity_id']
            uttr_dict["doc_id"]=target_list[i]['doc_id']
            uttr_dict["type"]='title'

        elif i-1 in target_list:
            uttr_dict["domain"]=target_list[i-1]['domain']
            uttr_dict["entity_id"]=target_list[i-1]['entity_id']
            uttr_dict["doc_id"]=target_list[i-1]['doc_id']
            uttr_dict["type"]='body'
    
        domain_session.append(uttr_dict)
    
    target_list={}
    domain_session_logs.append(domain_session)

    return domain_session_logs



domain_session_logs=find_entire_log()
#with open('post-training_data/dstc9/domain_session_logs.json', 'w') as f:
#   json.dump(domain_session_logs[:100], f, indent=4)

def preprocess_dstc9(domain_session_logs):
   
    new_logs=[]
    new_labels=[]
    for dialog in tqdm(domain_session_logs):
        for i,uttr_dict in enumerate(dialog):
            if dialog[i]['speaker']=='S':
                
                temp_log=[]
                for j in range(i):
                    temp_log.append({'speaker':dialog[j]['speaker'],'text':dialog[j]['text']})
                new_logs.append(temp_log)
                
                if 'entity_id' in uttr_dict:
                    domain=uttr_dict['domain']
                    entity_id=uttr_dict['entity_id']
                    doc_id=uttr_dict['doc_id']
                    new_labels.append(
                        {
                            'target': True,
                            "knowledge": [
                                {
                                "domain": domain,
                                "entity_id": entity_id,
                                "doc_id":  doc_id
                                }
                            ],
                            "response": dialog[i]['text']
                        }
                    )
                else:
                    new_labels.append(
                        {
                            'target': True,
                            "response": dialog[i]['text']
                        }
                    )
    
    return new_logs,new_labels

dstc9_logs,dstc9_label=preprocess_dstc9(domain_session_logs)

with open('post-training_data/dstc9/logs.json', 'w') as f:
    json.dump(dstc9_logs, f, indent=4)
print("# of dstc9_logs %d"%(len(dstc9_logs)))
with open('post-training_data/dstc9/labels.json', 'w') as f:
    json.dump(dstc9_label, f, indent=4)
print("# of dstc9_label %d"%(len(dstc9_label)))

with open('post-training_data/dstc9/val/logs.json', 'w') as f:
    json.dump(dstc9_logs[-100:], f, indent=4)
with open('post-training_data/dstc9/val/labels.json', 'w') as f:
    json.dump(dstc9_label[-100:], f, indent=4)







print("END Program")

print("end")