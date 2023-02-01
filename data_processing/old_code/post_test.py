import json
from numpy.core.arrayprint import set_string_function
from tqdm import tqdm
import re
import numpy as np
import random
from normalize import english_cleaners



def post_normal(path):
    with open('pred/test/%s.json'%path, 'r') as f:
        labels = json.load(f)
    
    for i in range(len(labels)):
        if labels[i]['target']==True:
            labels[i]['response']=english_cleaners(labels[i]['response'])
    
    with open('pred/test/post/%s_post.json'%path, 'w') as f:
        json.dump(labels, f, indent=4)
    
post_normal('final_ks-f2_single-data_final_mix')
post_normal('final_ks-fam_single-data_final_mix')
post_normal('final_ks-f2faf1-data_final_mix')
post_normal('final_ks-famf2fa-data_final_mix')
post_normal('final_ks-famf2faf1-data_final_mix')


print("END Program")

print("end")