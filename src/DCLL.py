#The primary name we thougt is DCLLFA(DCLL), so in source code some function or variable use these name, but after we worte the paper, we decide change it to DLP-LoRA

import peft
import json
import torch
import torch.nn as nn
from src import train
from src import Process
from src import Get_data
from src import LoraModel
from src import ClassifierUtilities
from srcClassifier.ClassifierModel import Transformer_Classifier
from transformers import AutoModelForCausalLM, AutoTokenizer,AlbertTokenizer,AlbertConfig,RobertaTokenizer,RobertaConfig
from peft import PeftModel

def Normal_ModelLoader(model_pth):
    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    model = AutoModelForCausalLM.from_pretrained(
        model_pth,
        torch_dtype="auto",
        device_map="cuda:0",
    )
    return model,tokenizer

def ModelWithoutDCLL(max,model_pth,isTrain,Step,Train_data_path,Test_data_path,LoRA_pth,modeSave,cuda_mem,isTestUseLoRA = False):
    model,tokenizer = Normal_ModelLoader(model_pth)
    labels = ["input","output"]
    Train_datasets = Get_data.process_QADatasets_json(Train_data_path,labels,False)
    Test_datasets = Get_data.process_QADatasets_json(Test_data_path,labels,False)
    deltatime = 0
    if(isTrain == True):
        Train_datasets = Get_data.process_QADatasets_json(Train_data_path,labels,False)
        Test_datasets = Get_data.process_QADatasets_json(Test_data_path,labels,False)
        train.train(tokenizer,model,5e-5,Train_datasets,Test_datasets,LoRA_pth,Step,max,modeSave,cuda_mem,False)
    if(isTrain == False):
        seed = 12345  # 选择一个固定的种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        if(isTestUseLoRA == True):
            model = PeftModel.from_pretrained(model,LoRA_pth)
        deltatime = Process.test(Step,model,Test_datasets,tokenizer,max,modeSave)
    return deltatime

def DCLLFA_ModelLoader(file,model_pth,Classifier_pth,mode):
    if file is None:
        return []
    with open(file.name, 'r', encoding='utf-8') as f:
        pth_list = f.read().splitlines()
    typs = len(pth_list)
    DCLLFA = LoraModel.MutiLoRA(pth_list,model_pth)
    DCLLFA.collect_LoraParam()
    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    model = AutoModelForCausalLM.from_pretrained(
                model_pth,
                torch_dtype="auto",
                device_map="cuda:0",
            )
    model = peft.PeftModel.from_pretrained(model,pth_list[0])
    model = LoraModel.CausalFusionLM(model,DCLLFA).cuda()
    for param in model.parameters():
        param.requires_grad = False
    if(mode == "albert"):
        ClassifierTokenizer = AlbertTokenizer.from_pretrained("./BasicModel/Classifiers/albert-base-v2")
        ClassifierConfig = AlbertConfig.from_json_file("./BasicModel/Classifiers/albert-base-v2/config.json")
    elif(mode == "Roberta-base-distil"):
        ClassifierTokenizer = RobertaTokenizer.from_pretrained("./BasicModel/Classifiers/roberta-base-squad2-distilled")
        ClassifierConfig = RobertaConfig.from_json_file("./BasicModel/Classifiers/roberta-base-squad2-distilled/config.json")
    Classifier = Transformer_Classifier(ClassifierConfig,typs,mode).cuda()
    dic = torch.load(Classifier_pth)
    Classifier.load_state_dict(dic,strict=False)
    return model,tokenizer,Classifier,ClassifierTokenizer

def DCLLFA(pth_list,model_pth,Classifier_pth,Test_data_path,test_Step,mode,mx):
    """
    Classifier:分类器
    model:模型
    tokenizer:编码器
    ClassifierTokenizer:分类器编码器
    Test_data_pth:测试数据集
    test_Step:测试步数
    需要加载前置初始化函数
    """
    model,tokenizer,Classifier,ClassifierTokenizer = DCLLFA_ModelLoader(pth_list,model_pth,Classifier_pth,mode)
    seed = 12345  # 选择一个固定的种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    Test_datasets = Get_data.process_QADatasets_json(Test_data_path,["input","output"])
    deltaTime = Process.testMultiAdapter(Classifier,test_Step,model,Test_datasets,tokenizer,ClassifierTokenizer,mx)
    return deltaTime

def ClassifierLoader(typs,mode):
    if(mode == "albert"):
        ClassifierTokenizer = AlbertTokenizer.from_pretrained("./BasicModel/Classifiers/albert-base-v2")
        ClassifierConfig = AlbertConfig.from_json_file("./BasicModel/Classifiers/albert-base-v2/config.json")
    elif(mode == "Roberta-base-distil"):
        ClassifierTokenizer = RobertaTokenizer.from_pretrained("./BasicModel/Classifiers/roberta-base-squad2-distilled")
        ClassifierConfig = RobertaConfig.from_json_file("./BasicModel/Classifiers/roberta-base-squad2-distilled/config.json")
    Classifier = Transformer_Classifier(ClassifierConfig,typs,mode).cuda()
    return  Classifier,ClassifierTokenizer

def ClassifierTrainer(mode,typs,train_pth,test_pth,epoch,batch):
    Classifier,tokenizer = ClassifierLoader(typs,mode)
    ClassifierUtilities.train(Classifier,tokenizer,mode,typs,train_pth,test_pth,epoch,batch)
    return "Finish"

def jsonReader(file1,file2):
    with open(file1,"r") as r:
        file1 = json.load(r)
    with open(file2,"r") as r:
        file2 = json.load(r)
    for i in file2:
        file1.append(i)
    with open("./TestOutputs/Fusion/Fusion.json","w") as w:
        json.dump(file1, w, indent=4)
    return "Finish"

def jsonChanger(file1):
    with open(file1,"r") as r:
        file1 = json.load(r)
    fileName,_ = Process.get_next_filename("./TestOutputs/datasetChanger")
    for i in file1:
        i["output"] = _
    with open(fileName,"w") as w:
        json.dump(file1, w, indent=4)
    return "Finish"
