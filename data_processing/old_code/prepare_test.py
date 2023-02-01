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

with open('dstc10_test/logs.json', 'r') as f:
    test_log = json.load(f)

#label은 외부지식 사용할때만....

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


post_logs, post_labels=prepare_post(test_log)





with open('data_posttrain/test10_post/logs.json', 'w') as f:
    json.dump(post_logs, f, indent=4)
print("# of test_logs %d"%(len(post_logs)))
with open('data_posttrain/test10_post/labels.json', 'w') as f:
    json.dump(post_labels, f, indent=4)
print("# of test_labels %d"%(len(post_labels)))


#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")