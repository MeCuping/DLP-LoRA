import pandas
import json
import random
import os

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
        #loads可以把字符转成字典（格式正确的话),而且注意，是loads而不是load
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
        #因为句首已经有[CLS]所以去掉
        ans = b[i]
        bidata.append([ques,ans])
    print("获取数据集成功")
    return bidata

def MakingDatasets(pth,mode,labels,saveName,type):
    """
    mode
    parquet是parquet
    specialJ是指以并列dict存储的数据
    J是list[dict["input","output"]]
    type
    True是train
    False是test
    """
    if(mode == "parquet"):
        datasets = process_QADatasets_parquet(pth,labels)
    elif(mode == "specialJ"):
        datasets = process_QADatasets_json(pth,labels,True)
    elif(mode == "J"):
        datasets = process_QADatasets_json(pth,labels,False)
    else:
        raise ImportError("Error Mode")
    dics = []
    for i in datasets:
        dic = {"input":i[0],"output":i[1]}
        dics.append(dic)
    if(type == True):
        type = "train"
    else:
        type = "test"
    with open(f"./DataSets/DataSetsClassifier/TestLoraClassification/{type}/{saveName}.json","w") as writer:
        json.dump(dics,writer,indent=4)

def MakingClassificationDatasets(pthRoot,type):
    """
    type
    tain 训练
    test 测试
    """
    pths = [os.path.join(pthRoot, f) for f in os.listdir(pthRoot) if os.path.isfile(os.path.join(pthRoot, f))]
    datasets = []
    start = 0
    for i in pths:
        dataset = process_QADatasets_json(i,["input","output"])
        datasets.append(dataset)
        print(i,"\n数据总对数为",len(dataset))
    everyDatasets = int(input("请输入每个数据集取的数据数量"))
    if(type == "test"):
        start = int(input("输入单个训练集选取量，以选取测试集"))
    for j,i in enumerate(datasets):
        datasets[j] = i[start:start+everyDatasets]
    dics = []
    for m,i in enumerate(datasets):
        for j in i:
            dic = {"input":j[0],"output":m}
            dics.append(dic)
    random.shuffle(dics)
    with open(f"./DataSets/DataSetsClassifier/{type}_Every{everyDatasets}.json","w") as writer:
        json.dump(dics,writer,indent=4)

def RandomFusionTestDatasets(pthRoot,randomNum):
    pths = [os.path.join(pthRoot, f) for f in os.listdir(pthRoot) if os.path.isfile(os.path.join(pthRoot, f))]
    datasets = []
    dataFusionsets = []
    for i in pths:
        dataset = process_QADatasets_json(i,["input","output"])
        datasets.append(dataset)
        print(i,"\n数据总对数为",len(dataset))
    everyDatasets = int(input("请输入想要获取的测试融合数据数量"))
    for j,i in enumerate(datasets):
        datasets[j] = i[0:everyDatasets]
    for i in range(everyDatasets):
        randomSeed = []
        randomTask = []
        inp = ""
        outp = ""
        for j in range(randomNum):
            randomSeed.append(random.randint(0,everyDatasets-1))
            randomTask.append(random.randint(0,len(datasets)-1))
        for n,m in enumerate(randomTask):
            inp += datasets[m][randomSeed[n]][0]
            outp += datasets[m][randomSeed[n]][1]
        dataFusion = {"input":inp,"output":outp}
        dataFusionsets.append(dataFusion)
    with open(f"./DataSets/DataSetsClassifier/Fusion{everyDatasets}.json","w") as writer:
        json.dump(dataFusionsets,writer,indent=4)

def MakingAllDatasets(pthRoot,type):
    """
    type
    tain 训练
    test 测试
    """
    pths = [os.path.join(pthRoot, f) for f in os.listdir(pthRoot) if os.path.isfile(os.path.join(pthRoot, f))]
    datasets = []
    start = 0
    for i in pths:
        dataset = process_QADatasets_json(i,["input","output"])
        datasets.append(dataset)
        print(i,"\n数据总对数为",len(dataset))
    everyDatasets = int(input("请输入每个数据集取的数据数量"))
    if(type == "test"):
        start = int(input("输入单个训练集选取量，以选取测试集"))
    for j,i in enumerate(datasets):
        datasets[j] = i[start:start+everyDatasets]
    dics = []
    for m,i in enumerate(datasets):
        for j in i:
            dic = {"input":j[0],"output":j[1]}
            dics.append(dic)
    random.shuffle(dics)
    with open(f"./DataSets/DataSetsClassifier/{type}_Every{everyDatasets}.json","w") as writer:
        json.dump(dics,writer,indent=4)

if __name__ == "__main__":
    #需要手动将数据集转换成list[dict["input","output"]]形式
    """
    #STP1 制作可批处理的数据集
    MakingDatasets(pth = "./DataSets/LoRADataSets/mental_health_counseling_conversations.json",
                    mode = "specialJ",
                    labels = ["Context","Response"],
                    saveName = "MentalHealth",
                    type = True)
    """
    MakingAllDatasets("./DataSets/DataSetsClassifier/TestLoraClassification/train","train")
    #测试并生成融合数据然后测试
    #MakingClassificationDatasets("./DataSets/DataSetsClassifier/TestLoraClassification/train","test")
    #RandomFusionTestDatasets("./DataSets/DataSetsClassifier/TestLoraClassification/train",2)