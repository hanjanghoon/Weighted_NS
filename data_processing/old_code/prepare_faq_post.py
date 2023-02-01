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
random.seed(1228)
np.random.seed(1228)

seen_entity_num=1
num_k=10
min_count=20
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
labels9= train_labels9+val_labels9


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


#정답 맵핑하기.
valid_turn_cnt=0
entdoc2uttr={'title':{},'body':{}}
#정답 매칭 
for log,label in zip(logs9,labels9):
    if label['target']==True:
        
        if len(label['knowledge'])>1:
            print("wowowowwow")
        
        entity_id=label['knowledge'][0]['entity_id']
        doc_id=label['knowledge'][0]['doc_id']
        domain=label['knowledge'][0]['domain']

        if entity_id=='*':#taxi,train 일단 skip
            continue
        valid_turn_cnt+=1
        
        if (entity_id,doc_id) not in entdoc2uttr['title']:
            entdoc2uttr['title'][(entity_id,doc_id)]=[]

        if (entity_id,doc_id) not in entdoc2uttr['body']:
            entdoc2uttr['body'][(entity_id,doc_id)]=[]

        try:
            user_text=log[-1]['text'].lower()
        except:
            user_text=log[0]['text'].lower()
        
        system_text=label['response'].lower()

        entity_name=kb10[domain][str(entity_id)]['name'].lower()

        user_text=replace_entity(user_text,entity_name,domain)
        system_text=replace_entity(system_text,entity_name,domain)
        
        entdoc2uttr['title'][(entity_id,doc_id)].append(user_text)
        entdoc2uttr['body'][(entity_id,doc_id)].append(system_text)


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


#첫번재 dstc 9 to dstc 10
#cosine smiliarity

def mapping_kb9_kb10(mapping_dict,num_k):
    min_p=0.85
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    for domain in kb10:
        # Two lists of sentences
        if domain=='taxi' or domain=='train' or domain=='attraction':
            continue
        
        for type in mapping_dict:
            if type=='body':
                min_p=0.85

            kb9_faqs=kb9_domain_q_dict[domain][type]
            #문제는 atrraction 어케할거냐?
            if not kb9_faqs:#attraction 일단 빠짐.
                continue

            kb10_faqs=kb10_domain_q_dict[domain][type]

            #Compute embedding for both lists
            kb9_faq_embeddings = model.encode(kb9_faqs, convert_to_tensor=True)
            kb10_faq_embeddings = model.encode(kb10_faqs, convert_to_tensor=True)


            #첫번재 dstc 9 to dstc 10
            #Compute cosine-similarits
            cosine_scores = util.pytorch_cos_sim(kb9_faq_embeddings, kb10_faq_embeddings)
            topk_values_2d,topk_indices_2d=torch.topk(cosine_scores, k=num_k, dim=-1)
            for i,(topk_values,topk_indices) in enumerate(zip(topk_values_2d,topk_indices_2d)):
                '''
                if type=='body':
                    print("\nDSTC9\n%s \n"%kb9_faqs[i])
                    print("DSTC10")
                #jang
                if ('31','20')in kb9_mapping_dict[domain][type][i]:
                    print('here')
                '''
                for topk_val, topk_idx in zip(topk_values,topk_indices):#여기서 나중에 top k
                    if topk_val > min_p:
                        #if type=='body':
                            #print("%s \t %.3f"%(kb10_faqs[topk_idx],float(topk_val)))
                        
                        kb10_key_list=kb10_mapping_dict[domain][type][int(topk_idx)]
                        for key in kb10_key_list:
                            if key not in mapping_dict[type]:
                                mapping_dict[type][key]={"entdoc":[],"kb9":[],"kb10":[]}
                            mapping_dict[type][key]["entdoc"].extend(kb9_mapping_dict[domain][type][i])

                            idx_tuple=(domain,i)
                            if idx_tuple not in mapping_dict[type][key]["kb9"]:
                                mapping_dict[type][key]["kb9"].append(idx_tuple)
            
            #두번째 dstc10->dstc9
            cosine_scores_transpose=torch.transpose(cosine_scores, 0, 1)
            topk_values_2d,topk_indices_2d=torch.topk(cosine_scores_transpose, k=num_k, dim=-1)

            for i,(topk_values,topk_indices) in enumerate(zip(topk_values_2d,topk_indices_2d)):
            #print("DSTC10\n%s \n"%kb10_faqs[i])
            #print("DSTC9")
            #나중에 안본것 찾으려고 하는거얌
                for topk_val, topk_idx in zip(topk_values,topk_indices):#여기서 나중에 top k
                    #print("%s \t %.3f"%(kb9_faqs[topk_idx],float(topk_val)))
                    if topk_val > min_p:
                        kb10_key_list=kb10_mapping_dict[domain][type][i]
                        for key in kb10_key_list:
                            if key not in mapping_dict[type]:
                                mapping_dict[type][key]={"entdoc":[],"kb9":[],"kb10":[]}
                            #어자피 이거 묶음이라 원소 하나만 있어도됨. 그래서 앞 원소로 파악
                            if kb9_mapping_dict[domain][type][int(topk_idx)][0] in mapping_dict[type][key]['entdoc']:
                                break
                            else:
                                mapping_dict[type][key]['entdoc'].extend(kb9_mapping_dict[domain][type][int(topk_idx)])
                                #entitiy 없으면 바로 문장으로 넣어줄거임.
                                idx_tuple=(domain,int(topk_idx))
                                if idx_tuple not in mapping_dict[type][key]["kb9"]:
                                    mapping_dict[type][key]["kb9"].append(idx_tuple)   
        
        
    return mapping_dict

def mapping_kb10_kb10(mapping_dict):
    #세번째 dstc10 to dstc10

    min_p=0.8
    for domain in kb10:
        
        if domain=='taxi' or domain=='train':
            continue
        
        for type in mapping_dict:
            if type=='body':
                min_p=0.85

            kb10_faqs=kb10_domain_q_dict[domain][type]

            if not kb10_faqs:
                continue

            model = SentenceTransformer('paraphrase-mpnet-base-v2')
            kb10_faq_embeddings = model.encode(kb10_faqs, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(kb10_faq_embeddings,kb10_faq_embeddings)
            topk_values_2d,topk_indices_2d=torch.topk(cosine_scores, k=num_k, dim=-1)
            for i,(topk_values,topk_indices) in enumerate(zip(topk_values_2d,topk_indices_2d)):
                #print("\nDSTC10\n%s \n"%kb10_faqs[i])
                #print("DSTC10")

                for topk_val, topk_idx in zip(topk_values,topk_indices):#여기서 나중에 top k
                    #print("%s \t %.3f"%(kb10_faqs[topk_idx],float(topk_val)))
                    if topk_val > min_p:
                        kb10_key_list=kb10_mapping_dict[domain][type][i]
                        for key in kb10_key_list:

                            if key not in mapping_dict[type]:
                                mapping_dict[type][key]={"entdoc":[],"kb9":[],"kb10":[]}

                            idx_tuple=(domain,int(topk_idx))
                            if idx_tuple not in mapping_dict[type][key]["kb10"]:
                                mapping_dict[type][key]["kb10"].append(idx_tuple)

        
    return mapping_dict

def check_kb10_entity_link(mapping_dict):
    kb10_entitiy_length=len(kb10_every_entity)
    for type in mapping_dict :
        cnt=0
        cnt2=0
        average=0
        average2=0
        for (domain,entity_id,doc_id) in kb10_every_entity:
            if (entity_id,doc_id) in mapping_dict[type]:
                average+=len(mapping_dict[type][(entity_id,doc_id)]["entdoc"])+len(mapping_dict[type][(entity_id,doc_id)]["kb9"])+len(mapping_dict[type][(entity_id,doc_id)]["kb10"])
                cnt+=1
                if domain=='attraction':
                    average2+=len(mapping_dict[type][(entity_id,doc_id)]["entdoc"])+len(mapping_dict[type][(entity_id,doc_id)]["kb9"])+len(mapping_dict[type][(entity_id,doc_id)]["kb10"])
                    cnt2+=1
                #print(domain,entity_id,doc_id)
        if cnt2==0:
            cnt2=1
        print("\n%s\nseen entitiy : %d / %d\nseen attraction entitiy: %d\naverage_variation=%.3f\naverage attaction varation=%.3f"%(type,cnt,kb10_entitiy_length,cnt2,average/cnt,average2/cnt2))

def mapping_variation(mapping_dict,bart_gen,t5_gen,min_count):

    mapping_uttr2={'title':{},'body':{}}
    total_cnt=0
    attr_variation_cnt=0
    attr_cnt=0
    
    for (domain,entity_id,doc_id) in kb10_every_entity:
        
        


        if entity_id not in mapping_uttr2['title']:
            mapping_uttr2['title'][entity_id]={}

        if entity_id not in mapping_uttr2['body']:
            mapping_uttr2['body'][entity_id]={}    
        
      
        
        for type in mapping_dict:
            
            if type=='body':
                min_count=30
                

            if doc_id not in mapping_uttr2[type][entity_id]:
                mapping_uttr2[type][entity_id][doc_id]=[]

            variation_cnt=0
            #첫번째 mapping dict 사용 - dstc9 dialog
            for kb9_ent_doc in mapping_dict[type][(entity_id,doc_id)]['entdoc']:
                kb9_ent_doc= tuple(map(int,kb9_ent_doc))
                if kb9_ent_doc in entdoc2uttr[type]:
                    dstc9_log_list=entdoc2uttr[type][kb9_ent_doc]
                    mapping_uttr2[type][entity_id][doc_id].extend(dstc9_log_list)
                    variation_cnt+=len(dstc9_log_list)
            
            #print("\ndstc9_log=%d"%variation_cnt)
            

            if variation_cnt<min_count and type=='title': 
                if entity_id in bart_gen[type]:
                    if doc_id in bart_gen[type][entity_id]:
                        variation_list=bart_gen[type][entity_id][doc_id]
                        mapping_uttr2[type][entity_id][doc_id].extend(variation_list)
                        variation_cnt+=len(variation_list)
                #print("\nbart=%d"%variation_cnt)
            
            #t5
            if variation_cnt<min_count and type=='title':
                if entity_id in t5_gen[type]:
                    if doc_id in t5_gen[type][entity_id]:
                        variation_list=t5_gen[type][entity_id][doc_id]
                        mapping_uttr2[type][entity_id][doc_id].extend(variation_list)
                        variation_cnt+=len(variation_list)
                #print("\nt5=%d"%variation_cnt)
            
            #그다음은 3개이하면 넣어줌
            #dstc9 faq 사용
            if variation_cnt<min_count:
                for domain,idx in mapping_dict[type][(entity_id,doc_id)]['kb9']:
                    mapping_uttr2[type][entity_id][doc_id].append(kb9_domain_q_dict[domain][type][idx])
                    variation_cnt+=1
                #print("\ndstc9_faq=%d"%variation_cnt)
            #dstc10 faq사용        
            if variation_cnt<min_count:
                for domain,idx in mapping_dict[type][(entity_id,doc_id)]['kb10']:
                    mapping_uttr2[type][entity_id][doc_id].append(kb10_domain_q_dict[domain][type][idx])
                    variation_cnt+=1
                #print("\ndstc10_faq=%d"%variation_cnt)
            #나머지
            if variation_cnt<min_count:
                print("cautious pair (%d %d) = %d"%(int(entity_id),int(doc_id),variation_cnt))
            #if not mapping_uttr2[kb10_entity[0]][kb10_entity[1]]:
                #del mapping_uttr2[kb10_entity[0]][kb10_entity[1]]
            
            total_cnt+=variation_cnt
            
            if domain=='attraction':
                attr_variation_cnt+=variation_cnt
                attr_cnt+=1
            #print(cnt)
    print("\n%s\n최종 variation average = %.2f\n"%(type,total_cnt/(2*len(kb10_every_entity))))
    print("attr variation average = %.2f\n"%(attr_variation_cnt/attr_cnt))
    return mapping_uttr2


mapping_dict={'title':{},'body':{}}
#kb9 kb10mapping
mapping_dict=mapping_kb9_kb10(mapping_dict,num_k)
check_kb10_entity_link(mapping_dict)

#마지막 없다면 kb10
mapping_dict=mapping_kb10_kb10(mapping_dict)
check_kb10_entity_link(mapping_dict)
#참고로 attraction 230개

with open('faq_generation/bart_gen_pro.json', 'r') as f:
    bart_gen = json.load(f)
with open('faq_generation/t5_gen_pro.json', 'r') as f:
    t5_gen = json.load(f)

mapping_uttr2=mapping_variation(mapping_dict,bart_gen,t5_gen,min_count)


#원래는 대화세션당 모든 엔티티를 다 넣으려고 했으나 대화가 너무 많아져서 3개만 넣는다.
#넣기 위해서 knowledg entity index를 설정한다.



def replace_domain(text,domain,entity_name):
    
    text=copy.deepcopy(text.lower())
    entity_name=entity_name.lower()
    if "__%s__"%domain in text:
        text=text.replace("__%s__"%domain,entity_name)

    text=text.replace('cambridge','san francisco')
    text=' '.join(text.split())#공백 제거
    text=text.replace(" ?","?")#공백 제거
    text=text.strip()
    
    return text

#넘으면 나중에 생각...
debug=0
new_logs=[]
new_labels=[]
for (domain,entity_id,doc_id) in tqdm(kb10_every_entity):
    #돌리면서 넘겨줌..
    if domain=='taxi' or domain=='train':
        continue
    title=kb10[domain][entity_id]['docs'][doc_id]['title']
    body=kb10[domain][entity_id]['docs'][doc_id]['body']

    entity_name=kb10[domain][entity_id]['name']

    if mapping_uttr2['title'][entity_id][doc_id]:
            user=random.sample(mapping_uttr2['title'][entity_id][doc_id],1)[0].lower()
    else:
        user=title

    if mapping_uttr2['body'][entity_id][doc_id]:
            system=random.sample(mapping_uttr2['body'][entity_id][doc_id],1)[0].lower()
    else:
        system=body
    user=replace_domain(user,domain,entity_name)
    system=replace_domain(system,domain,entity_name)
    
    new_logs.append([{'speaker':'U','text':user}])
    new_labels.append({
        'target': True,
        "knowledge": [
            {
            "domain": domain,
            "entity_id": entity_id,
            "doc_id": doc_id
            }
        ],
        "response": system
        })
    #print(selected_uttr_pos)
    #갯수중에 복원 추출로 랜덤하게 3개를 뽑는다



with open('data_posttrain/faq_post/logs.json', 'w') as f:
    json.dump(new_logs, f, indent=4)
print("# of new_logs %d"%(len(new_logs)))
with open('data_posttrain/faq_post/labels.json', 'w') as f:
    json.dump(new_labels, f, indent=4)
print("# of new_labels %d"%(len(new_labels)))


#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")