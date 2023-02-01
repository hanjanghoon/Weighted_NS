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



#hotel,restaurant,attraction
#issue! train, texi를 날릴것인가? 일단 도메인이 다르기 때문에 날린다.
with open('dstc10_data/dstc10_knowledge.json', 'r') as f:
    kb10 = json.load(f)


#전체 대화는 71348
with open('multiwoz/dials.json', 'r') as f:
    mwoz = json.load(f)



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

                new_labels.append(
                    {
                        'target': True,
                        "response": dialog[i]['text']
                    }
                )

    return new_logs,new_labels

mwoz_logs,mwoz_label=preprocess_dstc9(mwoz)

try:
    if not os.path.exists("post-training_data/mwoz"):
            os.makedirs("post-training_data/mwoz")
except OSError:
    print ('Error: Creating directory. ' +  "post-training_data/mwoz")

with open('post-training_data/mwoz/logs.json', 'w') as f:
    json.dump(mwoz_logs, f, indent=4)
print("# of mwoz_logs %d"%(len(mwoz_logs)))
with open('post-training_data/mwoz/labels.json', 'w') as f:
    json.dump(mwoz_label, f, indent=4)
print("# of mwoz_label %d"%(len(mwoz_label)))





print("END Program")

print("end")