import json

#전체 대화는 71000
with open('data_posttrain/dstc9/logs.json', 'r') as f:
    dstc9_logs = json.load(f)
with open('data_posttrain/dstc9/labels.json', 'r') as f:
    dstc9_labels = json.load(f)

#전체 대화는 49000
with open('data_final/train/logs.json', 'r') as f:
    dstc10_logs = json.load(f)
with open('data_final/train/labels.json', 'r') as f:
    dstc10_labels = json.load(f)

#대화는 70000
with open('data_posttrain/mwoz/logs.json', 'r') as f:
    mwoz_logs = json.load(f)
with open('data_posttrain/mwoz/labels.json', 'r') as f:
    mwoz_labels = json.load(f)

#전체 대화는 10000
with open('data_posttrain/faq_post/logs.json', 'r') as f:
    faq_logs= json.load(f)
with open('data_posttrain/faq_post/labels.json', 'r') as f:
    faq_labels = json.load(f)

#전체 대화는 700
with open('data_posttrain/val10/post_logs.json', 'r') as f:
    val10_logs = json.load(f)
with open('data_posttrain/val10/post_labels.json', 'r') as f:
    val10_labels = json.load(f)



#with open('data_posttrain/rule_aug/logs.json', 'r') as f:
#    rule_logs = json.load(f)
#with open('data_posttrain/rule_aug/labels.json', 'r') as f:
#    rule_labels = json.load(f)





'''
with open('data_posttrain/val10/val10_testlogs.json', 'r') as f:
    val10_testlogs = json.load(f)
with open('data_posttrain/val10/val10_testlabels.json', 'r') as f:
    val10_testlabels = json.load(f)
'''

with open('data_posttrain/test10_post/logs.json', 'r') as f:
    test_logs = json.load(f)
with open('data_posttrain/test10_post/pred_labels.json', 'r') as f:
    pred_labels = json.load(f)
with open('data_posttrain/test10_post/labels.json', 'r') as f:
    test_labels = json.load(f)

aggregate_logs=dstc9_logs+dstc10_logs+mwoz_logs+faq_logs+val10_logs*5
aggregate_labels= dstc9_labels+dstc10_labels+mwoz_labels+faq_labels+val10_labels*5




valid_test_logs=[]
valid_test_labels=[]

for i in range(len(test_logs)):
    if pred_labels[i]['target']==True:
        valid_test_logs.append(test_logs[i])
        valid_test_labels.append({'target':True,'response':test_labels[i]['response']})
        '''
        print()
        for uttr_dict in test_logs[i]:
            print('[Text]=\t',uttr_dict['text'])
        print(test_labels[i]['response'])
        print()
        '''
test_logs=valid_test_logs
test_labels=valid_test_labels


with open('data_posttrain/aggregate/train/logs.json', 'w') as f:
    json.dump(aggregate_logs, f, indent=4)
print("# of aggregate_logs %d"%(len(aggregate_logs)))

with open('data_posttrain/aggregate/train/labels.json', 'w') as f:
    json.dump(aggregate_labels, f, indent=4)
print("# of aggregate_labels %d"%(len(aggregate_labels)))

#리얼 validation
with open('dstc10_data/val_logs.json', 'r') as f:
    val_logs = json.load(f)
with open('dstc10_data/val_labels.json', 'r') as f:
    val_labels = json.load(f)


print("# of val_logs %d"%(len(test_logs)))

with open('data_posttrain/aggregate/val/logs.json', 'w') as f:
    #json.dump(val10_testlogs, f, indent=4)
    json.dump(test_logs, f, indent=4)

with open('data_posttrain/aggregate/val/labels.json', 'w') as f:
    #json.dump(val10_testlabels, f, indent=4)
    json.dump(test_labels, f, indent=4)




print("END Program")

print("end")