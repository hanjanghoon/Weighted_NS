import json
from numpy.core.arrayprint import set_string_function
from tqdm import tqdm
import re
import numpy as np
import random
import copy
import torch
import os
random.seed(1228)
np.random.seed(1228)



with open('data_detection/train/logs.json', 'r') as f:
    train_logs2= json.load(f)
with open('data_final/train/logs.json', 'r') as f:
    train_logs3= json.load(f)

with open('data_final_asr/train/logs.json', 'r') as f:
    train_logs4= json.load(f)

with open('data_final_asr_mix/train/logs.json', 'r') as f:
    train_logs5= json.load(f)

with open('data_final_mix/train/logs.json', 'r') as f:
    train_logs6= json.load(f)





with open('dstc10_data/val_logs.json', 'r') as f:
    val_logs= json.load(f)

#label은 외부지식 사용할때만....
with open('dstc10_data/val_labels.json', 'r') as f:
    val_labels= json.load(f)

#final shuffle and split
val_logs=np.array(val_logs)
val_labels=np.array(val_labels)

idx = np.arange(len(val_logs))
np.random.shuffle(idx)

final_logs=val_logs[idx]
final_labels=val_labels[idx]

val_logs=val_logs.tolist()
val_labels=val_labels.tolist()

final_train_logs=val_logs[:63]
final_train_labels=val_labels[:63]

final_val_logs=val_logs[63:]
final_val_labels=val_labels[63:]

with open('data_split_val_paper/gen/train/logs.json', 'w') as f:
    json.dump(final_train_logs, f, indent=4)
with open('data_split_val_paper/gen/train/labels.json', 'w') as f:
    json.dump(final_train_labels, f, indent=4)

with open('data_split_val_paper/gen/val/logs.json', 'w') as f:
    json.dump(final_val_logs, f, indent=4)
with open('data_split_val_paper/gen/val/labels.json', 'w') as f:
    json.dump(final_val_labels, f, indent=4)


val_logs10 = final_train_logs
val_labels10 = final_train_labels

val_tuple=[]
for log,label in zip(val_logs10,val_labels10):
    val_tuple.append((log,label))

val_tuple=sorted(val_tuple,key=lambda x : (x[0][0]['text'],len(x[0])))


#test_tuple=val_tuple[232:]
#val_tuple=val_tuple[:232]


def find_entire_log(val_tuple):
    target_list={}
    previous=None
    dialog_cnt=0
    mixed_domain_cnt=0
    domain_session_logs=[]
    last_target=False
    for dialog,dialabel in tqdm(val_tuple,total=len(val_tuple[0])):

        #다음 대화 세션으로 넘어왓다는뜻.
        if previous:
            if previous[0]['text']!=dialog[0]['text']:
                if last_target==True:
                    print('last_append')
                    previous.append({'speaker':'S','text':last_response})

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
        previous=dialog
        

        
        if dialabel['target']:
            for k_dict in dialabel['knowledge']:
                target_list[len(dialog)-1]=k_dict
            last_target=True
            last_response=dialabel['response']
        else:
            last_target=False
            last_response=None

    domain_session=[]
    if last_target==True:
        print('last_append')
        previous.append({'speaker':'S','text':last_response})


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

def prepare_post(domain_session_logs):
   
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



domain_session_logs=find_entire_log(val_tuple)
post_logs,post_labels=prepare_post(domain_session_logs)





with open('data_posttrain/split_val10/post_logs.json', 'w') as f:
    json.dump(post_logs, f, indent=4)
print("# of post_logs %d"%(len(post_logs)))
with open('data_posttrain/split_val10/post_labels.json', 'w') as f:
    json.dump(post_labels, f, indent=4)
print("# of post_labels %d"%(len(post_labels)))

'''
val10_testlogs=[]
val10_testlabels=[]
for (log,label) in test_tuple:
    val10_testlogs.append(log)
    val10_testlabels.append(label)

with open('data_posttrain/val10/val10_testlogs.json', 'w') as f:
    json.dump(val10_testlogs, f, indent=4)
print("# of val10_testlogs %d"%(len(val10_testlogs)))

with open('data_posttrain/val10/val10_testlabels.json', 'w') as f:
    json.dump(val10_testlabels, f, indent=4)
print("# of test_tuple %d"%(len(val10_testlabels)))
'''
#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")