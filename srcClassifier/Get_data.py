import random
from transformers import BertTokenizer
from typing import Tuple
import json

def get_QAdataset(datasets:Tuple[Tuple[str,str]])->Tuple[str,str]:
    j = random.randint(0,len(datasets)-1)
    return datasets[j]

def process_QADatasets_json(dir,special = False):
    print("开始获取数据集")
    a = []
    b = []
    with open(dir,'r',encoding='utf-8') as r:
        if(special == True):
            dics = r.readlines()
        elif(special == False):
            dics = json.load(r)
    for j,i in enumerate(dics):
        if(special == True):
            m = json.loads("".join(i.split('\n')))
            a.append(m['input'])
            b.append(m['output'])
        elif(special == False):
            a.append(i['input'])
            b.append(i['output'])
    bidata = []
    for i in range(len(a)):
        ques = a[i]#+"A: [CLS] [MASK] "
        #因为句首已经有[CLS]所以去掉
        ans = b[i]
        bidata.append([ques,ans])
    print("获取数据集成功")
    return bidata

def process_QADatasets(dir):
    print("开始获取数据集")
    dataset = []
    datasets = []
    with open(dir,'r',encoding='utf-8') as r:
        dataset = r.readlines()
        for i in dataset:
            datasets.append(i.replace('\n',''))
    if(len(datasets)%2 == 1):
        datasets.pop()
    a = [i for j,i in enumerate(datasets) if j%2 == 0]
    b = [i for j,i in enumerate(datasets) if j%2 == 1]
    bidata = []
    for i in range(len(a)):
        ques = a[i]#+"A: [CLS] [MASK] "
        #因为句首已经有[CLS]所以去掉
        ans = b[i]
        bidata.append([ques,ans])
    random.shuffle(bidata)
    print("获取数据集成功")
    return bidata

def deal_inputs_batched_FIXED(tokenizer,bidata,commands):
    #第一个MASK是我要保存的
    messages_Q = [
        {"role": "system", "content": commands},
        {"role": "user", "content": bidata[0]}
    ]
    messages_A = [
        {"role": "system", "content": commands},
        {"role": "user", "content": bidata[0]},
        {"role": "assistant", "content": bidata[1]}
    ]
    input = tokenizer.apply_chat_template(
        messages_Q,
        tokenize=True,
        add_generation_prompt=True
    )
    label_total = tokenizer.apply_chat_template(
        messages_A,
        tokenize=True,
        add_generation_prompt=False
    )
    start_idx = len(input)
    start_idx_copy = start_idx-1
    sp = int(len(label_total)-len(input)+1)
    for i in range(sp):
        #加入一个无关紧要的占位符,防止超限
        input.append("_")
        input[start_idx] = label_total[start_idx-1]
        start_idx += 1
    return input[:-1],label_total,start_idx_copy

def deal_inputs_batched_developed(tokenizer,bidata,pad,commands):
    #第一个MASK是我要保存的
    messages_Q = [
        {"role": "system", "content": commands},
        {"role": "user", "content": bidata[0]}
    ]
    messages_A = [
        {"role": "system", "content": commands},
        {"role": "user", "content": bidata[0]},
        {"role": "assistant", "content": bidata[1]}
    ]
    input = tokenizer.apply_chat_template(
        messages_Q,
        tokenize=True,
        add_generation_prompt=True
    )
    label_total = tokenizer.apply_chat_template(
        messages_A,
        tokenize=True,
        add_generation_prompt=False
    )
    start_idx = len(input)
    start_idx_copy = start_idx-1
    sp = int((len(label_total)-len(input)+1)/pad)
    iptdatas = []
    lbldatas = []
    for i in range(sp):
        iptdatas.append(input[:])
        for j in range(pad):
            idx = start_idx+j
            #加入一个无关紧要的占位符,防止超限
            input.append("_")
            input[idx] = label_total[idx-1]
        start_idx += pad
        lbldatas.append(input[:])
    return iptdatas,lbldatas,start_idx_copy

def form_batch(batch_size,datasets):
    """
    好像长度不一致没法组成batch，后面试试用截断能不能解决
    """
    inp = None
    for i in range(batch_size):
        data = get_QAdataset(datasets)
        deal_inputs_batched_FIXED(data)