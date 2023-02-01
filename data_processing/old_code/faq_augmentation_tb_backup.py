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
seen_entity_num=2
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
    min_p=0.8
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    for domain in kb10:
        # Two lists of sentences
        if domain=='taxi' or domain=='train' or domain=='attraction':
            continue
        
        for type in mapping_dict:
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
            
                #print("\nDSTC9\n%s \n"%kb9_faqs[i])
                #print("DSTC10")

                for topk_val, topk_idx in zip(topk_values,topk_indices):#여기서 나중에 top k
                    #print("%s \t %.3f"%(kb10_faqs[topk_idx],float(topk_val)))
                    if topk_val > min_p:
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

    num_k=20
    min_p=0.7
    for domain in kb10:
        
        if domain=='taxi' or domain=='train':
            continue
        
        for type in mapping_dict:
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
            if variation_cnt<min_count:
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



#비슷한 것으로 찾을때는 entitiy를 넣어주는게 낫지 않을까?
#여기부터 faq 끼우기

with open('mydata/%s/domain_session_all.json'%mode, 'r') as f:
    domain_session = json.load(f)


def make_single_faq_dialog(dialog,faqs,entity_idx,doc_idx_list,selected_uttr_pos,domain,entity_name):
    dialog=copy.deepcopy(dialog)
    selected_uttr_pos=sorted(selected_uttr_pos)
    #before=dialog[:]
    offset = 0
    #faq 데이터 발화형식으로 변환 해주기
    uttr_faq_list=[]
    for i in doc_idx_list:
        user_faq_dict={}
        user_faq_dict['speaker']='U'
        #user_faq_dict['text']=faqs[str(i)]['title']
        user_faq_dict['text']=random.sample(mapping_uttr2['title'][entity_idx][i],1)[0].lower()

        user_faq_dict['domain']=domain+'_faq'
        user_faq_dict['entity_id']=entity_idx
        user_faq_dict['doc_id']=str(i)

        sys_faq_dict={}
        sys_faq_dict['speaker']='S'

        if mapping_uttr2['body'][entity_idx][i]:
            sys_faq_dict['text']=random.sample(mapping_uttr2['body'][entity_idx][i],1)[0].lower()
        else:
            sys_faq_dict['text']=faqs[str(i)]['body']

        sys_faq_dict['domain']=domain+'_faq'
        sys_faq_dict['entity_id']=entity_idx
        sys_faq_dict['doc_id']=str(i)
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
    return dialog 


def make_cross_faq_dialog(dialog,new_knowledge,cross_domain_dict):
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
            entity_name=new_knowledge[domain][entity_idx]['name']
            cross_faq_dict[domain]['entity_name']=entity_name

            uttr_faq_list=[]
            for i in cross_domain_dict[domain]['doc_idx_list']:
                user_faq_dict={}
                user_faq_dict['speaker']='U'
                #user_faq_dict['text']=hotel_faqs[str(i)]['title']
                user_faq_dict['text']=random.sample(mapping_uttr2['title'][entity_idx][i],1)[0].lower()
                user_faq_dict['domain']=domain+'_faq'
                user_faq_dict['entity_id']=entity_idx
                user_faq_dict['doc_id']=str(i)

                sys_faq_dict={}
                sys_faq_dict['speaker']='S'
                if mapping_uttr2['body'][entity_idx][i]:
                    sys_faq_dict['text']=random.sample(mapping_uttr2['body'][entity_idx][i],1)[0].lower()
                else:
                    sys_faq_dict['text']=faqs[str(i)]['body']
                sys_faq_dict['domain']=domain+'_faq'
                sys_faq_dict['entity_id']=entity_idx
                sys_faq_dict['doc_id']=str(i)
                
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

    return dialog 


#여기서 이제 넣을 거임.
with open('mydata/new_knowledge.json', 'r') as f:
    new_knowledge = json.load(f)


augmented_domain_session_logs=[]
#원래는 대화세션당 모든 엔티티를 다 넣으려고 했으나 대화가 너무 많아져서 3개만 넣는다.
#넣기 위해서 knowledg entity index를 설정한다.



entities_num={
    "hotel":len(new_knowledge['hotel']),
    "restaurant": len(new_knowledge['restaurant']),
    "attraction": len(new_knowledge['attraction'])
}
entities_list={
    "hotel":list(new_knowledge['hotel'].keys()),
    "restaurant": list(new_knowledge['restaurant'].keys()),
    "attraction": list(new_knowledge['attraction'].keys())
}

entity_list_idx={
    "hotel":0,
    "restaurant": 0,
    "attraction": 0
}

#매번 모든 엔티티를 볼경우 셔플링을 해준다.
print("domain_session_length=%d\n"%len(domain_session))

for dialog in tqdm(domain_session):
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
        augmented_domain_session_logs.append(dialog)
        continue
    

    if cross_flag:
        debug=0
        for _ in range(seen_entity_num):#식당이 더 많아서 이걸로 봄. 아니다 위에서 추출해서 맞춰놓음.
            #entity의 faq 개수를 본다.
            cross_domain_dict={'hotel':{},'restaurant':{},'attraction':{}}
            for domain in possible_loc:
                if possible_loc[domain] :
                    while True:
                        entity_idx=entities_list[domain][entity_list_idx[domain]]
                        entity_list_idx[domain]+=1
                        if entity_list_idx[domain]>=entities_num[domain]:
                            random.shuffle(entities_list[domain])
                            entity_list_idx[domain]=0
                        if entity_idx in mapping_uttr2['title']:
                            break

                    faqs=new_knowledge[domain][entity_idx]['docs']
                    entity_name=new_knowledge[domain][entity_idx]['name']

                    candidates_doc=list(mapping_uttr2['title'][entity_idx].keys())
                    num_faqs=len(candidates_doc)

                    #섞으나 안섞으나 똑같음.
                    #순서대로 2개의 질문을 뽑고
                    
                    doc_idx_list=random.sample(candidates_doc,min(2,len(candidates_doc)))
                    selected_pos=np.random.choice(possible_loc[domain],min(2,len(candidates_doc)))
                    
                    cross_domain_dict[domain]['entity_idx']=entity_idx
                    cross_domain_dict[domain]['doc_idx_list']=doc_idx_list
                    cross_domain_dict[domain]['selected_pos']=selected_pos

            faq_dialog=make_cross_faq_dialog(dialog,new_knowledge,cross_domain_dict)
            augmented_domain_session_logs.append(faq_dialog)
    else:
        #넘으면 나중에 생각...
        debug=0
        for domain in possible_loc:
            if possible_loc[domain]:
                for _ in range(seen_entity_num):
                    #돌리면서 넘겨줌..
                    while True:
                        entity_idx=entities_list[domain][entity_list_idx[domain]]
                        entity_list_idx[domain]+=1
                        if entity_list_idx[domain]>=entities_num[domain]:
                            random.shuffle(entities_list[domain])
                            entity_list_idx[domain]=0
                        if entity_idx in mapping_uttr2['title']:
                            break
                    
                    faqs=new_knowledge[domain][entity_idx]['docs']
                    entity_name=new_knowledge[domain][entity_idx]['name']

                    candidates_doc=list(mapping_uttr2['title'][entity_idx].keys())
                    num_faqs=len(candidates_doc)

                    #섞으나 안섞으나 똑같음.
                    #순서대로 2개의 질문을 뽑고
                    
                    doc_idx_list=random.sample(candidates_doc,min(3,len(candidates_doc)))
                    selected_pos=np.random.choice(possible_loc[domain],min(3,len(candidates_doc)))

                    #print(selected_uttr_pos)
                    #갯수중에 복원 추출로 랜덤하게 3개를 뽑는다.
                    faq_dialog=make_single_faq_dialog(dialog,faqs,entity_idx,doc_idx_list,selected_pos,domain,entity_name)
                    augmented_domain_session_logs.append(faq_dialog)
        
      
        #else : unc_transport
print("augmented dialog 수 : %d"%(len(augmented_domain_session_logs)))
#entitiy=3 doc=all -> 15만개.
#enitity=3 doc=1 ->2만


try:
    if not os.path.exists("mydata/%s"%version):
            os.makedirs("mydata/%s"%version)
except OSError:
    print ('Error: Creating directory. ' +  "mydata/newlog%d"%version)


with open('mydata/%s/augmented_domain_session_logs.json'%version, 'w') as f:
    json.dump(augmented_domain_session_logs, f, indent=4)


#대화 세션 쪼개기.
with open('mydata/%s/augmented_domain_session_logs.json'%version, 'r') as f:
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





with open('mydata/%s/logs.json'%version, 'w') as f:
    json.dump(new_logs, f, indent=4)
print("# of new_logs %d"%(len(new_logs)))
with open('mydata/%s/labels.json'%version, 'w') as f:
    json.dump(new_labels, f, indent=4)
print("# of new_labels %d"%(len(new_labels)))


#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")