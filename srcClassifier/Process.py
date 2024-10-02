from srcClassifier import evaluation
import torch

def to_one_hot(data,Classification_num):
    data_zero = torch.zeros([len(data),Classification_num]).cuda()
    for j,i in enumerate(data):
        data_zero[j][i] = 1
    return data_zero


def tokenizing(bidata,tokenizer,typs):
    input = bidata[0]
    label = to_one_hot(bidata[1],typs)
    batch_encoding = tokenizer.batch_encode_plus(
        input,
        padding=True,               # 填充至最长序列
        truncation=True,            # 截断超出最大长度的序列
        return_tensors='pt',        # 返回 PyTorch 张量
        add_special_tokens=False,   # 添加 [CLS] 和 [SEP] tokens
        max_length=256              # 设置上下文上限（超出截断)
    )
    input_ids = batch_encoding['input_ids'].cuda()
    attention_mask = batch_encoding['attention_mask'].cuda()
    return (input_ids,attention_mask,label)

def test(TestDataset,classifier,test_batch,test_num,tokenizer,typs):
    classifier.eval()
    ac = 0
    for i in range(test_num):
        data = TestDataset.Random_get(test_batch)
        inp,attn,label = tokenizing(data,tokenizer,typs)
        out = classifier(inp,attn)
        ac += evaluation.Ac_test(out,data[1])
    print("测试AC:",ac/(test_batch*test_num))
    classifier.train()