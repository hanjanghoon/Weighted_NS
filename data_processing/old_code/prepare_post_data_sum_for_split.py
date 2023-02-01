import json

#전체 대화는 71000
with open('data_posttrain/dstc9/logs.json', 'r') as f:
    dstc9_logs = json.load(f)
with open('data_posttrain/dstc9/labels.json', 'r') as f:
    dstc9_labels = json.load(f)

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

with open('data_posttrain/split_val10/post_logs.json', 'r') as f:
    split_post_val_logs=json.load(f)
with open('data_posttrain/split_val10/post_labels.json', 'r') as f:
    split_post_val_labels=json.load(f)


#전체 대화는 49000
with open('data_final/train/logs.json', 'r') as f:
    dstc10_logs = json.load(f)
with open('data_final/train/labels.json', 'r') as f:
    dstc10_labels = json.load(f)

aggregate_logs=dstc9_logs+dstc10_logs+mwoz_logs+faq_logs+split_post_val_logs*20
aggregate_labels= dstc9_labels+dstc10_labels+mwoz_labels+faq_labels+split_post_val_labels*20


with open('data_posttrain/aggregate_paper/train/logs.json', 'w') as f:
    json.dump(aggregate_logs, f, indent=4)
print("# of aggregate_logs %d"%(len(aggregate_logs)))

with open('data_posttrain/aggregate_paper/train/labels.json', 'w') as f:
    json.dump(aggregate_labels, f, indent=4)
print("# of aggregate_labels %d"%(len(aggregate_labels)))

with open('data_split_val_paper/gen/val/logs.json', 'r') as f:
    val_logs = json.load(f)
with open('data_split_val_paper/gen/val/labels.json', 'r') as f:
    val_labels = json.load(f)

with open('data_posttrain/aggregate_paper/val/logs.json', 'w') as f:
    #json.dump(val10_testlogs, f, indent=4)
    json.dump(val_logs, f, indent=4)

with open('data_posttrain/aggregate_paper/val/labels.json', 'w') as f:
    #json.dump(val10_testlabels, f, indent=4)
    json.dump(val_labels, f, indent=4)






#mix generation 용
with open('data_split_val_paper/gen/train/logs.json', 'r') as f:
    split_val_train_logs = json.load(f)
with open('data_split_val_paper/gen/train/labels.json', 'r') as f:
    split_val_train_labels = json.load(f)

paper_train_logs=split_val_train_logs*10+dstc10_logs
paper_train_labels=split_val_train_labels*10+dstc10_labels

with open('data_final_mix_paper/train/logs.json', 'w') as f:
    json.dump(paper_train_logs, f, indent=4)
with open('data_final_mix_paper/train/labels.json', 'w') as f:
    json.dump(paper_train_labels, f, indent=4)

with open('data_final_mix_paper/val/logs.json', 'w') as f:
    #json.dump(val10_testlogs, f, indent=4)
    json.dump(val_logs, f, indent=4)
with open('data_final_mix_paper/val/labels.json', 'w') as f:
    #json.dump(val10_testlabels, f, indent=4)
    json.dump(val_labels, f, indent=4)

print("END Program")

print("end")