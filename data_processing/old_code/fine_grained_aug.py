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
random.seed(713)
np.random.seed(713)

version="trainf_v2"
mode="train"
seen_entity_num=1
num_k=5
min_count=10
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




#hotel,restaurant,attraction
#issue! train, texi를 날릴것인가? 일단 도메인이 다르기 때문에 날린다.
with open('dstc10_data/dstc10_knowledge.json', 'r') as f:
    kb10 = json.load(f)

#전체 대화는 71348
with open('data/%s/logs.json'%mode, 'r') as f:
    logs9 = json.load(f)

#label은 외부지식 사용할때만....
with open('data/%s/labels.json'%mode, 'r') as f:
    labels9 = json.load(f)

def replace_entity(text,entity_name,domain):
    
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


def insert_faq(text,domain_q_dict,map_dict,domain,type):

    if text in domain_q_dict[domain][type]:
        #이미 들어가 있으면 그 인덱스의 정보에다가 entitiy doc id 추가함.
        map_dict[domain][type][domain_q_dict[domain][type].index(text)].append((entity_id,doc_id))
    else:
        #한번도 안들어 왓으면
        #기존 길이의 인덱스로 값이 들어가고 거기에 맵핑
        map_dict[domain][type][len(domain_q_dict[domain][type])]=[(entity_id,doc_id)]
        domain_q_dict[domain][type].append(text)
    
    return domain_q_dict, map_dict

db10list=[]
kb10list=[]
kb9list=[]
#dstc9 entity를 domain 별로 나눠보자
kb10_domain_q_dict={}
kb9_domain_q_dict={}
kb10_mapping_dict={}
kb9_mapping_dict={}
cnt=0
kb10_every_entity=[]

for domain in kb10:
    if domain=='taxi' or domain=='train':
        continue
    kb9_cnt=0
    kb10_cnt=0
    kb9_domain_q_dict[domain]={"title":[],"body":[]}
    kb10_domain_q_dict[domain]={"title":[],"body":[]}
    kb9_mapping_dict[domain]={"title":{},"body":{}}
    kb10_mapping_dict[domain]={"title":{},"body":{}}
    no_log_cnt=0
    
    for entity_id in kb10[domain]:
        
        entity_name=kb10[domain][entity_id]['name'].lower()
        kb10list.append(entity_name)
        #도메인별 엔티티 채워줌
        #kb10에서 나타난 모든 발화를 저장함
        #일단 질문으로만 가보자.
        for doc_id in kb10[domain][entity_id]['docs']:
            title=kb10[domain][entity_id]['docs'][doc_id]['title'].lower()
            body=kb10[domain][entity_id]['docs'][doc_id]['body'].lower()
            
            title=replace_entity(title,entity_name,domain)
            body=replace_entity(body,entity_name,domain)
            
            if kb10[domain][entity_id]['city']=="San Francisco" :
                kb10_every_entity.append((domain,entity_id,doc_id))
                kb10_cnt+=1

                kb10_domain_q_dict,kb10_mapping_dict=insert_faq(title,kb10_domain_q_dict,kb10_mapping_dict,domain,'title')
                kb10_domain_q_dict,kb10_mapping_dict=insert_faq(body,kb10_domain_q_dict,kb10_mapping_dict,domain,'body')
                
            else:
                kb9_cnt+=1

                #나중에 빼줘야 하니까 그리고 다시 넣어줘야 하네.
                #이경우는 대화 자체가 한번도 안나타난것.

                kb9_domain_q_dict,kb9_mapping_dict=insert_faq(title,kb9_domain_q_dict,kb9_mapping_dict,domain,'title')
                kb9_domain_q_dict,kb9_mapping_dict=insert_faq(body,kb9_domain_q_dict,kb9_mapping_dict,domain,'body')
                    
                
            #print("None")
    print("kb9 domain=%s unique title=%d unique body=%d all query=%d"%(domain,len(kb9_domain_q_dict[domain]['title']),len(kb9_domain_q_dict[domain]['body']),kb9_cnt))
    print("kb10 domain=%s unique title=%d unique body=%d all query=%d"%(domain,len(kb10_domain_q_dict[domain]['title']),len(kb10_domain_q_dict[domain]['body']),kb10_cnt))
    print()
    #hotel dialog = 3300, restaurnat 3300, attraction 1100
    #dialog 당 가용한 faq hotel = 8000, res=8000 attraction:2500
    #faq수  hotel=2000/3000 restauran=2800/5500 attraction=100/200
    #variation이 들어갈 경우..

print("end mapping & query list")




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




def replace_entity2entity(text,entity_name9,entity_name10,domain):
    
    text=copy.deepcopy(text.lower())
    entity_name9=entity_name9.lower()
    entity_name10=entity_name10.lower()

    if entity_name9 in text:
        text=text.replace(entity_name9,entity_name10)
    else:
        if domain!='attraction':
            for vari_enitity in kb9_entity_variation[domain]:
                if vari_enitity in text:
                    text=text.replace(vari_enitity,entity_name10)
                    break
    text=' '.join(text.split())#공백 제거
    text=text.replace(" ?","?")#공백 제거
    text=text.strip()
    
    return text

def make_fine_grained_log(domain_session_logs):
   
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    kb10_faq_emb={'hotel':{'title':[],'body':[]},'restaurant':{'title':[],'body':[]}}
    for domain in kb10_faq_emb:
        for type in kb10_faq_emb[domain]:
            kb10_faq_emb[domain][type]=model.encode(kb10_domain_q_dict[domain][type],convert_to_tensor=True)
    
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    new_logs=[]
    new_labels=[]
    cnt=0
    for dialog in tqdm(domain_session_logs):
        for i,uttr_dict in enumerate(dialog):
            if 'domain' in uttr_dict:
                domain=uttr_dict['domain']
                
                if domain=='train' or domain=="taxi":
                    break
                
                entity_id9=uttr_dict['entity_id']
                doc_id9=uttr_dict['doc_id']
                type=uttr_dict["type"]

                entity_name9=kb10[domain][str(entity_id9)]['name'].lower()
                text=replace_entity(uttr_dict['text'], entity_name9, domain)

                uttr_emb=model.encode(text, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(uttr_emb,kb10_faq_emb[domain][type])
                topk_values_2d,topk_indices_2d=torch.topk(cosine_scores, k=50, dim=-1)

                #print("\nDSTC9\n%s \n"%text)
                #print("DSTC10")

                if type=='title':
                    temp_pair={'title':[],'body':[]}

                for topk_val, topk_idx in zip(topk_values_2d[0],topk_indices_2d[0]):#여기서 나중에 top k
                    #print("%s \t %.3f"%(kb10_domain_q_dict[domain][type][int(topk_idx)],float(topk_val)))
                    if topk_val > 0.6:
                        kb10_entity_list=kb10_mapping_dict[domain][type][int(topk_idx)]
                        temp_pair[type].extend(kb10_entity_list)
                
                if type=='body':
                    inter_pair_list=[]
                    for title_pair in temp_pair['title']:
                        if title_pair in temp_pair['body']:
                            inter_pair_list.append(title_pair)
                    if not inter_pair_list:
                        print('error')
                        cnt+=1
                        continue
                    
                    entity_id10=inter_pair_list[0][0]
                    doc_id10=inter_pair_list[0][1]
                    entity_name10=kb10[domain][str(entity_id10)]['name'].lower()
                    
                    temp_log=[]
                    for j in range(i):
                        text=replace_entity2entity(dialog[j]['text'],entity_name9,entity_name10,domain)
                        temp_log.append({'speaker':dialog[j]['speaker'],'text':text})
                    new_logs.append(temp_log)
                    
                    new_labels.append(
                        {
                            'target': True,
                            "knowledge": [
                                {
                                "domain": domain,
                                "entity_id": entity_id10,
                                "doc_id":  doc_id10
                                }
                            ],
                            "response": replace_entity2entity(uttr_dict['text'],entity_name9,entity_name10,domain)
                        }
                    )
                   
    print("error=%d"%cnt)
    return new_logs,new_labels
new_logs,new_labels=make_fine_grained_log(domain_session_logs)






with open('gen_data/train/logs.json', 'w') as f:
    json.dump(new_logs, f, indent=4)
print("# of new_logs %d"%(len(new_logs)))
with open('gen_data/train/labels.json', 'w') as f:
    json.dump(new_labels, f, indent=4)
print("# of new_labels %d"%(len(new_labels)))

with open('gen_data/val/logs.json', 'w') as f:
    json.dump(new_logs[:100], f, indent=4)
print("# of new_logs %d"%(len(new_logs)))
with open('gen_data/val/labels.json', 'w') as f:
    json.dump(new_labels[:100], f, indent=4)


print("END Program")

print("end")