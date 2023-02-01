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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="15"


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

def insert_faq(text,domain_q_dict,map_dict,domain):

    if text in domain_q_dict[domain]:
        #이미 들어가 있으면 그 인덱스의 정보에다가 entitiy doc id 추가함.
        map_dict[domain][domain_q_dict[domain].index(text)].append((entity_id,doc_id))
    else:
        #한번도 안들어 왓으면
        #기존 길이의 인덱스로 값이 들어가고 거기에 맵핑
        map_dict[domain][len(domain_q_dict[domain])]=[(entity_id,doc_id)]
        domain_q_dict[domain].append(text)
    
    return domain_q_dict, map_dict

db10list=[]
kb10list=[]
kb9list=[]
#dstc9 entity를 domain 별로 나눠보자
kb10_domain_q_dict={}
kb10_mapping_dict={}
cnt=0
kb10_every_entity=[]

for domain in kb10:
    if domain=='taxi' or domain=='train':
        continue
    kb10_cnt=0
    kb10_domain_q_dict[domain]=[]
    kb10_mapping_dict[domain]={}
    no_log_cnt=0
    
    for entity_id in kb10[domain]:
        
        entity_name=kb10[domain][entity_id]['name'].lower()
        kb10list.append(entity_name)
        #도메인별 엔티티 채워줌
        #kb10에서 나타난 모든 발화를 저장함
        #일단 질문으로만 가보자.
        for doc_id in kb10[domain][entity_id]['docs']:
            title=kb10[domain][entity_id]['docs'][doc_id]['title'].lower()
            title=replace_entity(title,entity_name,domain)
            
            
            if kb10[domain][entity_id]['city']=="San Francisco" :
                kb10_every_entity.append((domain,entity_id,doc_id))
                kb10_cnt+=1

                kb10_domain_q_dict,kb10_mapping_dict=insert_faq(title,kb10_domain_q_dict,kb10_mapping_dict,domain)
                
                
                    
    print("kb10 domain=%s unique title=%d all query=%d"%(domain,len(kb10_domain_q_dict[domain]),kb10_cnt))
    print()
    #hotel dialog = 3300, restaurnat 3300, attraction 1100
    #dialog 당 가용한 faq hotel = 8000, res=8000 attraction:2500
    #faq수  hotel=2000/3000 restauran=2800/5500 attraction=100/200
    #variation이 들어갈 경우..

print("end mapping & query list")


#첫번재 dstc 9 to dstc 10
#cosine smiliarity


def mapping_kb10_kb10(mapping_dict):
    #세번째 dstc10 to dstc10

    num_k=50
    min_p=0.85
    for domain in kb10:
        
        if domain=='taxi' or domain=='train':
            continue
        
        kb10_faqs=kb10_domain_q_dict[domain]

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
                    kb10_key_list=kb10_mapping_dict[domain][i]
                    for key in kb10_key_list:

                        if key not in mapping_dict:
                            mapping_dict[key]=[]

                        idx_tuple=(domain,int(topk_idx))
                        if idx_tuple not in mapping_dict[key]:
                            mapping_dict[key].append(idx_tuple)

        
    return mapping_dict

def mapping_variation(mapping_dict):

    smil_knowledge={}
    total_cnt=0

    
    for (domain,entity_id,doc_id) in kb10_every_entity:
        
        key="%s__%s__%s"%(domain,entity_id,doc_id)
        if key not in smil_knowledge:
            smil_knowledge[key]=[]
    
        for _ ,idx in mapping_dict[(entity_id,doc_id)]:

            for ventity_id,vdoc_id in kb10_mapping_dict[domain][idx]:
                value="%s__%s__%s"%(domain,ventity_id,vdoc_id)
                if value==key:
                    continue
                smil_knowledge[key].append(value)
    
            
            #print(cnt)
    return smil_knowledge


mapping_dict={}
#kb9 kb10mapping

mapping_dict=mapping_kb10_kb10(mapping_dict)

smil_knowledge=mapping_variation(mapping_dict)

# with open('smil_knowledge.json', 'w') as f:
    # json.dump(smil_knowledge, f, indent=4)
print("# of smil_knowledge %d"%(len(smil_knowledge)))


#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")