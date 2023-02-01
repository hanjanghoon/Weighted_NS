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
with open('rule_aug/synthetic-hotel.json', 'r') as f:
    hotel = json.load(f)

with open('rule_aug/synthetic-restaurant.json', 'r') as f:
    restaurant = json.load(f)

with open('rule_aug/synthetic-attraction.json', 'r') as f:
    attraction = json.load(f)

rule_logs=[]
rule_label=[]
for dialog in tqdm(hotel+restaurant+attraction):
    if len(dialog)<2:
        continue
    else:
        temp_log=[]
        for turn in dialog['turns']:
            temp_log.append({'speaker':'S','text':turn['system']})
            temp_log.append({'speaker':'U','text':turn['user']})
        temp_log=temp_log[1:-1]    
        if temp_log:
            rule_logs.append(temp_log[:-1])
            rule_label.append(
                    {
                        'target': True,
                        "response": temp_log[-1]['text']
                    }
            )


with open('post-training_data/rule_aug/logs.json', 'w') as f:
    json.dump(rule_logs, f, indent=4)
print("# of rule_logs %d"%(len(rule_logs)))
with open('post-training_data/rule_aug/labels.json', 'w') as f:
    json.dump(rule_label, f, indent=4)
print("# of rule_label %d"%(len(rule_label)))





print("END Program")

print("end")