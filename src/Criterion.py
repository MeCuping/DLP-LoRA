import torch
import torch.nn as nn
import torch.nn.functional as F

#All In one
def AIO(mode,out,label,loss = None,weight = [1,1,1,1]):
    """
    loss如果不止处理一次那么可以使用继承
    请注意，在使用dkl时保证所有值为正(起码>1e-10)，否则log(x)x<=0将会没有输出
    weight对应
    0:Ce
    1:Dkl
    2:Cos
    3:Mse
    无论是否用到都请输入三位数列
    """
    vault = 1e-8
    sm = nn.Softmax(-1)
    mse = nn.MSELoss()
    label = label.view(-1,label.size(-1))
    out = out.view(-1,out.size(-1))
    if loss == None:
        loss = [0,0,0,0]
        if(mode == "Mse"):
            loss[3] = mse(out,label)*weight[3]
        if(mode == "Ce"):
            CE = nn.CrossEntropyLoss()
            loss[0] = CE(out,label)*weight[0]
        if(mode == "Dkl"):
            out = sm(out)
            loss[1] = F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')*weight[1]
        if(mode == "Cos"):
            loss[2] = F.cosine_similarity(out,label)*weight[2]
        if(mode == "CeDkl"):
            CE = nn.CrossEntropyLoss()
            loss[0] = (CE(out,label)*weight[0])
            out = sm(out)
            kl = F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')
            loss[1] = (kl*weight[1])
        if(mode == "CosCeDkl"):
            cos = F.cosine_similarity(out,label)
            CE = nn.CrossEntropyLoss()
            loss[0] = (CE(out,label)*weight[0])
            out = sm(out)
            kl = F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')
            loss[1] = (kl*weight[1])
            loss[2] = (cos*weight[2])
    elif(loss != None):
        if(mode == "Mse"):
            loss[3] += mse(out,label)*weight[3]
        if(mode == "Ce"):
            CE = nn.CrossEntropyLoss()
            loss[0] += CE(out,label)*weight[0]
        if(mode == "Dkl"):
            out = sm(out)
            loss[1] += F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')*weight[1]
        if(mode == "Cos"):
            loss[2] += F.cosine_similarity(out,label)*weight[2]
        if(mode == "CeDkl"):
            CE = nn.CrossEntropyLoss()
            loss[0] += CE(out,label)*weight[0]
            out = sm(out)
            kl = F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')
            loss[1] += kl*weight[1]
        if(mode == "CosCeDkl"):
            cos = F.cosine_similarity(out,label)
            CE = nn.CrossEntropyLoss()
            loss[0] += CE(out,label)*weight[0]
            out = sm(out)
            kl = F.kl_div(torch.log(label+vault),out+vault,reduction='batchmean')*weight[1]
            loss[1] += kl*weight[1]
            if(len(loss) == 3):
                loss[2] += cos*weight[2]
            else:
                loss.append(cos*weight[2])
    return loss


