import pandas
import random
import json

class Preprocess:
    def __init__(self,pth,*args) -> None:
        self.bidata_all = []
        self.pth = pth
        self.tuple = args
        self.bidata_processed = []

    def process_QADatasets_json(self):
        a = []
        b = []
        with open(self.pth,'r',encoding='utf-8') as r:
                dics = json.load(r)
        for j,i in enumerate(dics):
            a.append(i['input'])
            b.append(i['output'])
        bidata = []
        for i in range(len(a)):
            ques = a[i]#+"A: [CLS] [MASK] "
            ans = b[i]
            bidata.append([ques,ans])
        self.bidata_all = bidata

    def dealDatas(self):
        for i in self.bidata_all:
            input = i[0]
            label = i[1]
            input = "[CLS]"+input+"[SEP]"
            self.bidata_processed.append([input,label])

    def Random_get(self,batch):
        data_inp = []
        data_label = []
        for i in range(batch):
            rand = random.randint(0,len(self.bidata_processed)-1)
            data_inp.append(self.bidata_processed[rand][0])
            data_label.append(self.bidata_processed[rand][1])
        return [data_inp,data_label]
    
    def forward(self):
        print("开始获取数据集")
        self.process_QADatasets_json()
        self.dealDatas()
        print("获取数据集成功")
