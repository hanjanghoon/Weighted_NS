import json
import numpy as np
np.random.seed=1228
from sklearn.model_selection import train_test_split


#전체 대화는 71000
with open('data_final/train/logs.json', 'r') as f:
    final_logs = json.load(f)
with open('data_final/train/labels.json', 'r') as f:
    final_labels = json.load(f)

with open('data_final_asr/train/logs.json', 'r') as f:
    asr_logs = json.load(f)
with open('data_final_asr/train/labels.json', 'r') as f:
    asr_labels = json.load(f)


with open('dstc10_data/val_logs.json', 'r') as f:
    val_logs= json.load(f)

#label은 외부지식 사용할때만....
with open('dstc10_data/val_labels.json', 'r') as f:
    val_labels= json.load(f)

#263
print("val9 length %d"%len(val_logs))

final_logs+=val_logs
final_labels+=val_labels

asr_logs+=val_logs
asr_labels+=val_labels



# final_train_logs,final_val_logs ,final_train_labels,final_val_labels=\
#     train_test_split(final_logs,final_labels,test_size=0.05)


#final shuffle and split
final_logs=np.array(final_logs)
final_labels=np.array(final_labels)

idx = np.arange(len(final_logs))
np.random.shuffle(idx)

final_logs=final_logs[idx]
final_labels=final_labels[idx]

final_logs=final_logs.tolist()
final_labels=final_labels.tolist()

final_train_logs=final_logs[:-250]
final_train_labels=final_labels[:-250]

final_val_logs=final_logs[-250:]
final_val_labels=final_labels[-250:]



#asr shuffle and split
asr_logs=np.array(asr_logs)
asr_labels=np.array(asr_labels)

idx = np.arange(len(asr_logs))
np.random.shuffle(idx)

asr_logs=asr_logs[idx]
asr_labels=asr_labels[idx]

asr_logs=asr_logs.tolist()
asr_labels=asr_labels.tolist()

asr_train_logs=asr_logs[:-250]
asr_train_labels=asr_labels[:-250]

asr_val_logs=asr_logs[-250:]
asr_val_labels=asr_labels[-250:]




with open('data_final_mix/train/logs.json', 'w') as f:
    json.dump(final_train_logs, f, indent=4)
with open('data_final_mix/train/labels.json', 'w') as f:
    json.dump(final_train_labels, f, indent=4)

with open('data_final_mix/val/logs.json', 'w') as f:
    json.dump(final_val_logs, f, indent=4)
with open('data_final_mix/val/labels.json', 'w') as f:
    json.dump(final_val_labels, f, indent=4)

with open('data_final_asr_mix/train/logs.json', 'w') as f:
    json.dump(asr_train_logs, f, indent=4)
with open('data_final_asr_mix/train/labels.json', 'w') as f:
    json.dump(asr_train_labels, f, indent=4)

with open('data_final_asr_mix/val/logs.json', 'w') as f:
    json.dump(asr_val_logs, f, indent=4)
with open('data_final_asr_mix/val/labels.json', 'w') as f:
    json.dump(asr_val_labels, f, indent=4)



print("END Program")

print("end")