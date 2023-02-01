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



#대화 세션 쪼개기.
with open('asr/dstc10asr_final.json', 'r') as f:
    augmented_domain_session_logs = json.load(f)

def random_sample(given_list,num):
    return_list=[]
    while True:
        picked_list=random.sample(given_list,min(len(given_list),num))
        num=num-len(picked_list)
        return_list.extend(picked_list)
        if num<=0:
            break
    return return_list

new_logs=[]
new_labels=[]
cnt=0
for dialog in tqdm(augmented_domain_session_logs):
    dialog_form=[]
    faq_uttr_idx_list=[]
    normal_uttr_idx_list=[]
    neg_idx=0
    for i in range(1,len(dialog)):
        if 'text_asr' in dialog[i]:
            dialog_form.append({'speaker':dialog[i]['speaker'],'text':dialog[i]['text_asr']})
        else:
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
                '''
                if dialog[0]=='cross_attraction':#이경우 겹치기 때문에 샘플링 주의.
                    if 'attraction' not in dialog[i]['domain']:
                        normal_uttr_idx_list.append(i)
                else:
                    normal_uttr_idx_list.append(i)
                '''
    #여기서 답3개 negative 세개 할건대
    #앞을 랜덤하게 자를까 말까? 처음부터 한다. vs 앞을 3턴정도로만 유지한다.
    #일단 안자름.
    #1. 답만들기 참고로 답은 하나의 대화당 3개의 entiy + 모든 faq (8쌍) 약 한개의 대화당 25개의 답 생성.
    for user_idx in faq_uttr_idx_list:
        try:
            start_idx=random.randrange(0,user_idx-1, 2)
        except:
            start_idx=0
        new_logs.append(dialog_form[start_idx:user_idx])
        new_labels.append({
        'target': True,
        "knowledge": [
            {
            "domain": dialog[user_idx]['domain'].rstrip('_faq'),
            "entity_id": dialog[user_idx]['entity_id'],
            "doc_id": dialog[user_idx]['doc_id']
            }
        ],
        "response": dialog_form[user_idx]['text']
        })
    
    if not faq_uttr_idx_list:
        #taxi 랑 train 그냥 스킵
        #print(dialog[0])
        cnt+=1
        continue
    #negative sample

    try:
        user_idx_list=random_sample(normal_uttr_idx_list,max(len(faq_uttr_idx_list),3))
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




with open('data_dstc10asr/train/logs.json', 'w') as f:
    json.dump(new_logs, f, indent=4)
print("# of new_logs %d"%(len(new_logs)))
with open('data_dstc10asr/train/labels.json', 'w') as f:
    json.dump(new_labels, f, indent=4)
print("# of new_labels %d"%(len(new_labels)))


#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")