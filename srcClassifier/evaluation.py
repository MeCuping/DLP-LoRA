import torch

def Ac_test(out,label):
    label = torch.tensor(label).cuda()
    max = torch.argmax(out,dim=-1)
    test = max == label
    #用to去转换类别
    sum = torch.sum(test.to(dtype=torch.int))
    return sum