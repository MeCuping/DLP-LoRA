from srcClassifier.GetDatasets import Preprocess
from srcClassifier import Trainer
import torch

#print(Classifier)
#data = Train_DataLoader.Random_get(10)
#trys = Process.tokenizing(data,tokenzier)
def train(Classifier,tokenizer,mode,typs,train_pth,test_pth,epoch,batch):
    Train_DataLoader = Preprocess(train_pth)
    Test_DataLoader = Preprocess(test_pth)
    Train_DataLoader.forward()
    Test_DataLoader.forward()   
    Trainer.train(
        epoch = epoch,
        batch = batch,
        lr = 1e-5,
        test_num = 500,
        test_batch = 10,
        classifier = Classifier,
        TrainDataInstance = Train_DataLoader,
        TestDataInstance = Test_DataLoader,
        tokenizer = tokenizer,
        mode = mode,
        typs = typs
    )

def ClassifierInf(inp,Classifier,tokenizer,mode,max,KorP):
    """
    mode:
    topp
    topk
    KorP:
    topk_num
    P
    """
    #"./model/ClassifierAlbert.pth"
    inp = tokenizer.encode(inp)
    if(len(inp)>max):
        #取前max个
        inp = inp[len(inp)-max:]
    inp = torch.tensor([inp]).cuda()
    out = Classifier(inp)
    if(mode == "topp"):
        top,weight = Classifier.ClassifierOutput(out,"topp",KorP)
    if(mode == "topk"):
        top,weight = Classifier.ClassifierOutput(out,"topk",KorP)
    return top,weight