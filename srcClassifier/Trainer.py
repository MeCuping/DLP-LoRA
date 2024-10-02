from srcClassifier import Process,evaluation
import torch
import torch.nn as nn

def train(epoch,batch,lr,test_num,test_batch,classifier,TrainDataInstance,TestDataInstance,tokenizer,mode,typs):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params=classifier.parameters(),lr=lr)
    sm = 0
    sm_loss = 0
    ac = 0
    for i in range(epoch):
        if(sm%500 == 0 and sm !=0):
            avg_loss = sm_loss/sm
            avg_ac = ac/(sm*batch)
            print(f"第{(i)*batch}个,平均loss:{avg_loss},平均AC:{avg_ac}")
            Process.test(TestDataInstance,classifier,test_batch,test_num,tokenizer,typs)
            sm = 0
            sm_loss = 0
            ac = 0
        bidata = TrainDataInstance.Random_get(batch)
        inpt,attn_mask,label = Process.tokenizing(bidata,tokenizer,typs)
        out = classifier(inpt,attn_mask)
        #以首位CLS做预测标签
        loss = criterion(out,label)
        ac += evaluation.Ac_test(out,bidata[1])
        sm += 1
        sm_loss += loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.empty_cache()
        if((i+1)%2500 == 0):
            dic = classifier.state_dict()
            torch.save(dic,f"./finish/Classifier/mode_{mode},lr_{lr},step_{i+1}.pth")