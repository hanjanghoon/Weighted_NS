import json
import numpy as np
np.random.seed=1228
from sklearn.model_selection import train_test_split


#전체 대화는 71000
with open('data_final/train/labels.json', 'r') as f:
    train_label = json.load(f)

#with open('data_final_mix/val/labels.json', 'r') as f:
    #val_label = json.load(f)


with open('data_final_asr/train/labels.json', 'r') as f:
    asr_train_label = json.load(f)

#with open('data_final_asr_mix/val/labels.json', 'r') as f:
    #asr_val_label = json.load(f)


for i in range(len(train_label)):
    if 'knowledge' in train_label[i]:
        train_label[i]['knowledge'][0]['entity_id']=int(train_label[i]['knowledge'][0]['entity_id'])
        train_label[i]['knowledge'][0]['doc_id']=int(train_label[i]['knowledge'][0]['doc_id'])

'''
for i in range(len(val_label)):
    if 'knowledge' in val_label[i]:
        val_label[i]['knowledge'][0]['entity_id']=int(val_label[i]['knowledge'][0]['entity_id'])
        val_label[i]['knowledge'][0]['doc_id']=int(val_label[i]['knowledge'][0]['doc_id'])
'''

for i in range(len(asr_train_label)):
    if 'knowledge' in asr_train_label[i]:
        asr_train_label[i]['knowledge'][0]['entity_id']=int(asr_train_label[i]['knowledge'][0]['entity_id'])
        asr_train_label[i]['knowledge'][0]['doc_id']=int(asr_train_label[i]['knowledge'][0]['doc_id'])
'''
for i in range(len(asr_val_label)):
    if 'knowledge' in asr_val_label[i]:
        asr_val_label[i]['knowledge'][0]['entity_id']=int(asr_val_label[i]['knowledge'][0]['entity_id'])
        asr_val_label[i]['knowledge'][0]['doc_id']=int(asr_val_label[i]['knowledge'][0]['doc_id'])
'''


with open('data_final/train/labels.json', 'w') as f:
    json.dump(train_label, f, indent=4)

#with open('data_final_mix/val/labels.json', 'w') as f:
#    json.dump(val_label, f, indent=4)

with open('data_final_asr/train/labels.json', 'w') as f:
    json.dump(asr_train_label, f, indent=4)

#with open('data_final_asr_mix/val/labels.json', 'w') as f:
#    json.dump(asr_val_label, f, indent=4)



print("END Program")

print("end")