import json
from tqdm import tqdm
import re,os

mode="train"
version="train"
domain_text={
            "hotel":["hotel","guesthouses","guest house","guesthouse"],
            "restaurant":["food","restaurant"," eat"]
            }
kb9_entity_variation={
    "hotel" : ['alexander','arbury','aylesbray','carolina','city centre north','express by holiday inn',
    'finches','hamilton','home from home','lovell','the cambridge belfry','a and b ', 'acorn ', 'alpha-milton',
    'arbury lodge guest', 'archway ', 'ashley ', 'autumn ', 'aylesbray lodge ', 'bridge ', 'gonville ', 
    'hobsons ', 'huntingdon marriott ', 'kirkwood ', 'leverton ', 'lime', 'the lensfield ', 'university arms ', 'warkworth ', 'worth'],
    "restaurant" : ['midsummer','alimentum','bloomsbury','shanghai family','shanghai','efes','meze','de luca cucina','chiquito restaurant',
    'chiquito','frankie and bennys','pizza hut cherry hinton','lucky star','gourmet burger kitchen','pizza hut city centre','one seven',
    'little seoul','shiraz','gandhi','sesame restaurant','sesame','fitzbillies','dojo','michaelhouse','ugly duckling','varsity',
    'stazione restaurant','stazione','jinling','peking','maharajah tandoori','cambridge lodge','hotpot','city stop','nirala','two two',
    'grafton','pipasha','zizzi','missing sock','cow pizza kitchen']
}
kb9_entity_variation['hotel']=sorted(kb9_entity_variation['hotel'],key=lambda x:len(x),reverse=True)
kb9_entity_variation['restaurant']=sorted(kb9_entity_variation['restaurant'],key=lambda x:len(x),reverse=True)
#hotel restaourant, taxi,train
with open('data/knowledge.json', 'r') as f:
    kb9 = json.load(f)
    
#hotel,restaurant,attraction
#issue! train, texi를 날릴것인가? 일단 도메인이 다르기 때문에 날린다.
with open('dstc10_data/dstc10_knowledge.json', 'r') as f:
    kb10 = json.load(f)

#전체 대화는 71348
with open('data/%s/logs.json'%mode, 'r') as f:
    logs9 = json.load(f)

#label은 외부지식 사용할때만....
with open('data/%s/labels.json'%mode, 'r') as f:
    labels9 = json.load(f)

kb10list=[]
kb9list=[]
#dstc9 entity를 domain 별로 나눠보자
kb9_domain_entity_dict={}
for domain in kb9:
    kb9_domain_entity_dict[domain]=[]
    for entity in kb9[domain]:
        if entity=='*':#train taxi
            continue
        else:
            kb9list.append(kb9[domain][entity]['name'].lower())
            #도메인별 엔티티 채워줌
            kb9_domain_entity_dict[domain].append(kb9[domain][entity]['name'].lower())


kb10_domain_entity_dict={}
for domain in kb10:
    kb10_domain_entity_dict[domain]=[]
    for entity in kb10[domain]:
        if entity=='*':#train taxi
            continue
        else:
            kb10list.append(kb10[domain][entity]['name'].lower())
            kb10_domain_entity_dict[domain].append(kb10[domain][entity]['name'].lower())



#dstc9 데이터에서 과연 특정 entitiy가 언급되는 대화쌍은 몇개일까? 그렇다면 대화는 모두 언급 되는가?
#18988 요걸 바꿔야함 약 1/4 이고 중복된 데이터 있음.




#과연 domain switching이 얼마나 많나? 총 7182개의 대화 세션 중 2670개임.
target_list={}
previous=None
dialog_cnt=0
mixed_domain_cnt=0
domain_session_logs=[]
for dialog,dialabel in tqdm(zip(logs9,labels9),total=len(logs9)):

    #다음 대화 세션으로 넘어왓다는뜻.
    if previous:
        if previous[0]['text']!=dialog[0]['text']:
            dialog_cnt+=1

            #previous가 타겟 대화 세션임.
            domain_session=[]
            #대화세션의 기본 도메인은 모른다임.
            previous_domain="uncertain"
            before_entity=False
            # 답이 없는 대화 세션이 있다?
            #if not domain:
             #   print('error')
            for i,uttr_dict in enumerate(previous):
                uttr_dict['text']=uttr_dict['text'].lower()
                #domain이 명시된 경우
                if i in target_list:
                    uttr_dict["domain"]=target_list[i]['domain']
                    before_entity=True
                    if target_list[i]['domain']=="taxi" or target_list[i]['domain']=="train":#기차랑 택시는 걍 넣어줌.
                        uttr_dict["entity_id"]=target_list[i]['entity_id']
                        uttr_dict["doc_id"]=target_list[i]['doc_id']
                        domain_session.append(uttr_dict)
                
                elif i-1 in target_list:
                    uttr_dict["domain"]=target_list[i-1]['domain']
                    before_entity=True
                    if target_list[i-1]['domain']=="taxi" or target_list[i-1]['domain']=="train":#기차랑 택시는 걍 넣어줌.
                        uttr_dict["entity_id"]=target_list[i-1]['entity_id']
                        uttr_dict["doc_id"]=target_list[i-1]['doc_id']
                        domain_session.append(uttr_dict)
                    
                else:
                    #entity로 찾기.
                    #도메인별 엔티티에 대한 여부 확인
                    domain_flag=False
                    for kb_domain in kb9_domain_entity_dict:
                        for entity in kb9_domain_entity_dict[kb_domain]:#식당 호텔 기차 택시
                            if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                                uttr_dict["domain"]=kb_domain
                                #entity는 __domain__으로 바꿔줌.
                                uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                                domain_session.append(uttr_dict)
                                
                                domain_flag=True
                                before_entity=True
                                previous_domain=kb_domain
                                break
                        if domain_flag==True:
                            break
                        #만약 생략된거나 변형된거라면? 변형은  b & b나 -이런것
                        if kb_domain not in kb9_entity_variation:
                            continue
                        for entity in kb9_entity_variation[kb_domain]:
                            entity=entity.strip()
                            if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                                if re.search(("\S"+entity),uttr_dict['text'].lower()) is not None:
                                    continue
                                uttr_dict["domain"]=kb_domain
                                #entity는 __domain__으로 바꿔줌.
                                uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                                domain_session.append(uttr_dict)
                                domain_flag=True
                                previous_domain=kb_domain
                                before_entity=True
                                break
                        if domain_flag==True:
                            break

                        
                    if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
                        continue
                    
                    #무효인 domain swithcin이 일어날 경우. 
                    diff_domain_flag=False

                    #텍스트에서 키워드를 통해 스위칭 되는걸 찾아보자 스위칭 될경우 도메인을 유지하면 안된다.
                    for txt_domain in domain_text:
                        for keyword in domain_text[txt_domain]:
                            if uttr_dict['text']=='okay what about any type of theatre? which is your favorite?':
                                print()
                            if keyword in uttr_dict['text'].lower():#before entity가 false는 entity가 등장하지 않았으면 도메인이 확실하여도 치환할수 없다.
                                if txt_domain!=previous_domain or before_entity==False:#이 경우 domain switching을 의미하지만 entity가 없으므로 특정 도메인이라고 해주긴 어렵다.
                                    uttr_dict["domain"]="uncertain"+"_"+txt_domain
                                    diff_domain_flag=True
                                else:
                                    uttr_dict["domain"]=previous_domain
                                    domain_flag=True
                                    domain_session.append(uttr_dict)
                                    break
                        if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
                            break
                    if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
                        continue
                    
                    #다 돌아도 모두다 다른 키워드일경우
                    if diff_domain_flag==True:    
                        domain_session.append(uttr_dict)
                        previous_domain=uttr_dict["domain"]
                        before_entity=False

                   
                    else:
                    #나머지의 경우 이전 domain을 따라감
                        uttr_dict["domain"]=previous_domain
                        domain_session.append(uttr_dict)
            
            target_list={}
            domain_session_logs.append(domain_session)


    previous=dialog
    
    #만약 도메인이 있다면. user-trun임.
    if dialabel['target']:
        for k_dict in dialabel['knowledge']:
            target_list[len(dialog)-1]=k_dict


dialog_cnt+=1
#이거는 crossdomain


#previous가 타겟 대화 세션임.
domain_session=[]
#대화세션의 기본 도메인은 모른다임.
previous_domain="uncertain"
before_entity=False
# 답이 없는 대화 세션이 있다?
#if not domain:
    #   print('error')
for i,uttr_dict in enumerate(previous):
    
    #domain이 명시된 경우
   #domain이 명시된 경우
    if i in target_list:
        uttr_dict["domain"]=target_list[i]['domain']
        before_entity=True
        if target_list[i]['domain']=="taxi" or target_list[i]['domain']=="train":#기차랑 택시는 걍 넣어줌.
            uttr_dict["entity_id"]=target_list[i]['entity_id']
            uttr_dict["doc_id"]=target_list[i]['doc_id']
            domain_session.append(uttr_dict)
    
    elif i-1 in target_list:
        uttr_dict["domain"]=target_list[i-1]['domain']
        before_entity=True
        if target_list[i-1]['domain']=="taxi" or target_list[i-1]['domain']=="train":#기차랑 택시는 걍 넣어줌.
            uttr_dict["entity_id"]=target_list[i-1]['entity_id']
            uttr_dict["doc_id"]=target_list[i-1]['doc_id']
            domain_session.append(uttr_dict)
    else:
        #entity로 찾기.
        #도메인별 엔티티에 대한 여부 확인
        domain_flag=False
        for kb_domain in kb9_domain_entity_dict:
            for entity in kb9_domain_entity_dict[kb_domain]:#식당 호텔 기차 택시
                if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                    uttr_dict["domain"]=kb_domain
                    #entity는 __domain__으로 바꿔줌.
                    uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                    domain_session.append(uttr_dict)
                    
                    domain_flag=True
                    before_entity=True
                    previous_domain=kb_domain
                    break
            if domain_flag==True:
                break
            #만약 생략된거나 변형된거라면? 변형은  b & b나 -이런것
            if kb_domain not in kb9_entity_variation:
                continue
            for entity in kb9_entity_variation[kb_domain]:
                entity=entity.strip()
                if entity in uttr_dict['text'].lower():#여기서 entity 전체가 없을수도 잇음 a and b 라고만 하기도 함. 요게 문제임.
                    if re.search(("\S"+entity),uttr_dict['text'].lower()) is not None:
                        continue
                    uttr_dict["domain"]=kb_domain
                    #entity는 __domain__으로 바꿔줌.
                    uttr_dict['text']=uttr_dict['text'].lower().replace(entity,"__"+kb_domain+"__")
                    domain_session.append(uttr_dict)
                    domain_flag=True
                    previous_domain=kb_domain
                    before_entity=True
                    break
            if domain_flag==True:
                break

            
        if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
            continue
        
        #무효인 domain swithcin이 일어날 경우. 
        diff_domain_flag=False

        #텍스트에서 키워드를 통해 스위칭 되는걸 찾아보자 스위칭 될경우 도메인을 유지하면 안된다.
        for txt_domain in domain_text:
            for keyword in domain_text[txt_domain]:
                if keyword in uttr_dict['text'].lower():#before entity가 false는 entity가 등장하지 않았으면 도메인이 확실하여도 치환할수 없다.
                    if txt_domain!=previous_domain or before_entity==False:#이 경우 domain switching을 의미하지만 entity가 없으므로 특정 도메인이라고 해주긴 어렵다.
                        uttr_dict["domain"]="uncertain"+"_"+txt_domain
                        diff_domain_flag=True
                    else:
                        uttr_dict["domain"]=previous_domain
                        domain_flag=True
                        domain_session.append(uttr_dict)
                        break
            if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
                break
        if domain_flag==True: # 다 됏으면 다음걸로 찾지 않는다.
            continue
        
        #다 돌아도 모두다 다른 키워드일경우
        if diff_domain_flag==True:    
            domain_session.append(uttr_dict)
            previous_domain=uttr_dict["domain"]
            before_entity=False

        
        else:
        #나머지의 경우 이전 domain을 따라감
            uttr_dict["domain"]=previous_domain 
            domain_session.append(uttr_dict)

target_list={}
domain_session_logs.append(domain_session)



#with open('mydata/domain_session_logs.json', 'w') as f:
 #   json.dump(domain_session_logs, f, indent=4)

#dialog session 별 수 확인 일단 호텔과 식당만 작업함.
h_cnt=0
r_cnt=0
hr_cnt=0
un_cnt=0
tr_cnt=0
for i,dialog in enumerate(domain_session_logs):
    h_flag=False
    r_flag=False
    for uttr_dict in dialog:
        if "__hotel__" in uttr_dict['text'].lower():
            h_flag=True
        elif "__restaurant__" in uttr_dict['text'].lower():
            r_flag=True
    
    if h_flag and r_flag:
        hr_cnt+=1
        domain_session_logs[i].insert(0,"cross")
    else:
        if h_flag:
            h_cnt+=1
            domain_session_logs[i].insert(0,"hotel")
        elif r_flag:
            r_cnt+=1
            domain_session_logs[i].insert(0,"restaurant")
        else:
            domain_set=set()
            for uttr_dict in dialog:
                if "uncertain_hotel" in uttr_dict['domain']:
                    domain_set.add("unc_hotel")
                elif "uncertain_restaurant" in uttr_dict['domain']:
                    domain_set.add("unc_restaurant")
            domain_set=sorted(domain_set)
            if len(domain_set)>1:
                domain_session_logs[i].insert(0,"unc_cross")
                un_cnt+=1
            elif len(domain_set)==1:
                domain_session_logs[i].insert(0,"".join(domain_set))
                un_cnt+=1
            else:
                domain_session_logs[i].insert(0,"unc_transport") #나중에 고려
                tr_cnt+=1


print("hrcnt: %d, hcnt:%d, rcnt:%d, uncnt:%d, trcnt:%d"%(hr_cnt,h_cnt,r_cnt,un_cnt,tr_cnt))
#hrcnt: 880, hcnt:2343, rcnt:2585, uncnt:775, trcnt:598



#일단 먼저 new_knowledge를 만듬. 기존 entity id doc id 유지 해야함.
new_knowledge={}
for domain in kb10:
    #idx=0
    new_knowledge[domain]={}
    for i in kb10[domain]:
        if domain=='taxi' or domain=='train':
            #new_knowledge[domain][i]=kb10[domain][i]
            continue
        if kb10[domain][i]['city'] == "San Francisco":
            new_knowledge[domain][i]=kb10[domain][i]

with open('mydata/new_knowledge.json', 'w') as f:
   json.dump(new_knowledge, f, indent=4)
  


#attraction multiwoz 추가

'''#에러수정
with open('multiwoz/attraction_cross_dialogue.json', 'r') as f:
    attract_cross = json.load(f)

attract_cross1=[]
for dialog in attract_cross:
    flag=True
    for uttr_dict in dialog[1:]:
        if "__restaurant__ __restaurant__" in uttr_dict['text']:
            flag=False
            break
        if "__attraction__ __attraction__" in uttr_dict['text']:
            flag=False
            break
        if "__hotel__ __hotel__" in uttr_dict['text']:
            flag=False
            break

    if flag:
        attract_cross1.append(dialog)


with open('multiwoz/attraction_cross_dialogue.json', 'w') as f:
    json.dump(attract_cross1, f, indent=4)


with open('multiwoz/attraction_only_dialogue.json', 'r') as f:
    attract_cross = json.load(f)

attract_cross1=[]
for dialog in attract_cross:
    flag=True
    for uttr_dict in dialog[1:]:
        if "__restaurant__ __restaurant__" in uttr_dict['text']:
            flag=False
            break
        if "__attraction__ __attraction__" in uttr_dict['text']:
            flag=False
            break
        if "__hotel__ __hotel__" in uttr_dict['text']:
            flag=False
            break
    if flag:
        attract_cross1.append(dialog)


with open('multiwoz/attraction_only_dialogue.json', 'w') as f:
    json.dump(attract_cross1, f, indent=4)
'''


with open('multiwoz/attraction_only_dialogue.json', 'r') as f:
    attract_single = json.load(f)

with open('multiwoz/attraction_cross_dialogue.json', 'r') as f:
    attract_cross = json.load(f)
            
if mode=='val':
    print("make_val_session!")
    attract_single=attract_single[-10:]
    attract_cross=attract_cross[-100:]
elif mode=='train':
    print("make_train_session!")
    attract_single=attract_single[:-10]
    attract_cross=attract_cross[:-100]

domain_session_all=[]
print("cross: %d, single:%d origin: %d"%(len(attract_cross),len(attract_single),len(domain_session_logs)))

for dialog in domain_session_logs+attract_single+attract_cross:
    domain_session_all.append(dialog)

try:
    if not os.path.exists("mydata/%s"%version):
        os.makedirs("mydata/%s"%version)
except OSError:
    print ('Error: Creating directory. ' +  "mydata/newlog%d"%version)

with open('mydata/%s/domain_session_all.json'%version, 'w') as f:
    json.dump(domain_session_all, f, indent=4)

'''

'''
#postprocess-generation
#bart
def gendata_process(file):
    with open(file, 'r') as f:
        faq_bart = json.load(f)
    cnt1=0
    cnt2=0
    total_cnt=0
    ent_doc_bart={'title':{},'body':{}}
    for domain in faq_bart:
        for entity_id in faq_bart[domain]:
            
            if entity_id=='*':
                continue

            ent_doc_bart['title'][entity_id]={}
            ent_doc_bart['body'][entity_id]={}

            entity_name=faq_bart[domain][entity_id]["name"].lower()
            for doc_id in faq_bart[domain][entity_id]["docs"]:
                    
                ent_doc_bart['title'][entity_id][doc_id]=[]
                ent_doc_bart['body'][entity_id][doc_id]=[]
                total_cnt+=1
                if 'title_paraphrase' not in faq_bart[domain][entity_id]["docs"][doc_id]:
                    #print(faq_bart[domain][entity_id]["docs"][doc_id]['title'])
                    cnt1+=1
                else:
                    
                    for gen_title in faq_bart[domain][entity_id]["docs"][doc_id]['title_paraphrase']:
                        gen_title=gen_title.lower()
                        gen_title=gen_title.strip()
                        gen_title=gen_title.replace('__domain__',domain)
                        if entity_name in gen_title:
                            gen_title=gen_title.replace(entity_name,'__%s__'%domain)
                        else:
                            for variation in kb9_entity_variation:
                                variation=variation.lower()
                                if variation in gen_title:
                                    gen_title=gen_title.replace(entity_name,'__%s__'%domain)
                                    break
                        if gen_title not in ent_doc_bart['title'][entity_id][doc_id]:
                            ent_doc_bart['title'][entity_id][doc_id].append(gen_title)
                
                if 'body_paraphrase' not in faq_bart[domain][entity_id]["docs"][doc_id]:
                    #print(faq_bart[domain][entity_id]["docs"][doc_id]['title'])
                    cnt2+=1
                else:
                    for gen_body in faq_bart[domain][entity_id]["docs"][doc_id]['body_paraphrase']:
                        gen_body=gen_body.lower()
                        gen_body=gen_body.strip()
                        gen_body=gen_body.replace('__domain__',domain)
                        if entity_name in gen_body:
                            gen_body=gen_body.replace(entity_name,'__%s__'%domain)
                        else:
                            for variation in kb9_entity_variation:
                                variation=variation.lower()
                                if variation in gen_body:
                                    gen_body=gen_body.replace(entity_name,'__%s__'%domain)
                                    break
                        if gen_body not in ent_doc_bart['body'][entity_id][doc_id]:
                            ent_doc_bart['body'][entity_id][doc_id].append(gen_body)
    
    print("title_excluded %d/%d"%(cnt1,total_cnt))
    print("body_excluded %d/%d"%(cnt2,total_cnt))        
    return ent_doc_bart
    
bart_gen=gendata_process('faq_generation/faq_bart_gen.json')
t5_gen=gendata_process('faq_generation/faq_t5_gen.json')
'''
with open('faq_generation/bart_gen_pro.json', 'w') as f:
    json.dump(bart_gen, f, indent=4)
with open('faq_generation/t5_gen_pro.json', 'w') as f:
    json.dump(t5_gen, f, indent=4)
'''

print("END Program")