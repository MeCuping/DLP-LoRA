import random
from transformers import BertTokenizer
import pandas
from typing import Tuple
import json

def TestDataset(datasets,i):
    return datasets[i]

def get_QAdataset(datasets:Tuple[Tuple[str,str]])->Tuple[str,str]:
    j = random.randint(0,len(datasets)-1)
    return datasets[j]

def process_QADatasets_parquet(dir,labels):
        bidata = []
        element1,element2 = labels
        reader = pandas.read_parquet(dir,engine='pyarrow')
        input = reader[element1].tolist()
        output = reader[element2].tolist()
        for j,i in enumerate(input):
            bidata.append([i,output[j]])
        return bidata

def process_QADatasets_json(dir,labels,special = False):
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
            a.append(m[labels[0]])
            b.append(m[labels[1]])
        elif(special == False):
            a.append(i[labels[0]])
            b.append(i[labels[1]])
    bidata = []
    for i in range(len(a)):
        ques = a[i]#+"A: [CLS] [MASK] "
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
        ans = b[i]
        bidata.append([ques,ans])
    random.shuffle(bidata)
    print("获取数据集成功")
    return bidata

def deal_inputs_batched_FIXED(tokenizer,bidata,commands,max):
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
    label = bidata[1]
    label_total = tokenizer.apply_chat_template(
        messages_A,
        tokenize=True,
        add_generation_prompt=False
    )
    start_idx = len(input)
    start_idx_copy = start_idx-1
    sp = int(len(label_total)-len(input)+1)
    for i in range(sp):
        input.append("_")
        input[start_idx] = label_total[start_idx-1]
        start_idx += 1
    if(len(input)>max):
        return input[:max],label_total[:max],start_idx_copy,label
    return input[:-1],label_total[:],start_idx_copy,label

def form_batch(batch_size,datasets):
    inp = None
    for i in range(batch_size):
        data = get_QAdataset(datasets)
        deal_inputs_batched_FIXED(data)