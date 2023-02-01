import json


#전체 대화는 71000
with open('data_final/train/logs.json', 'r') as f:
    final_logs = json.load(f)
with open('data_final/train/labels.json', 'r') as f:
    final_labels = json.load(f)

with open('data_final_asr/train/logs.json', 'r') as f:
    asr_logs = json.load(f)
with open('data_final_asr/train/labels.json', 'r') as f:
    asr_labels = json.load(f)



#대화는 70000
with open('data_eval/test/logs.json', 'r') as f:
    test9_logs = json.load(f)
with open('data_eval/test/labels.json', 'r') as f:
    test9_labels = json.load(f)


new_logs=[]
new_labels=[]
for log, label in zip(test9_logs,test9_labels):
    if 'sf' in label['source']:
        new_logs.append(log)
        new_labels.append(label)

print("test9 length %d"%len(new_logs))


final_logs+=new_logs
final_labels+=new_labels

asr_logs+=new_logs
asr_labels+=new_labels

'''
with open('data_final/train/logs.json', 'w') as f:
    json.dump(final_logs, f, indent=4)
with open('data_final/train/labels.json', 'w') as f:
    json.dump(final_labels, f, indent=4)
'''

with open('data_final_asr/train/logs.json', 'w') as f:
    json.dump(asr_logs, f, indent=4)
with open('data_final_asr/train/labels.json', 'w') as f:
     json.dump(asr_labels, f, indent=4)


print("END Program")

print("end")