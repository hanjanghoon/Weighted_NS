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
    logs = json.load(f)

#label은 외부지식 사용할때만....
with open('pred/test/post/assemble1.json', 'r') as f:
    labels = json.load(f)
print()
for log,label in zip(logs,labels):
    for log2 in log[-3:]:
        print("발화:",log2['text'])
    print()
    print("타겟:",label['target'])
    if label['target']:
        print("답변: ",label['response'])
        print("지식: ",label['knowledge'])
        
    print()

print("END Program")

print("end")