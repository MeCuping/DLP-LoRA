import torch
from collections import Counter
from src import Process

def Rouge_1_topp(inference,label):
    max_in_l = []
    max_la_l = []
    for i in range(inference.shape[1]):
        max_in = Process.top_p_decoding(inference[0][i][:]).item()
        max_in_l.append(max_in)
    for i in range(label.shape[1]):
        max_la = torch.argmax(label[0][i][:]).item()
        max_la_l.append(max_la)
    is_In = 0
    for i in max_in_l:
        if(i in max_la_l):
            is_In += 1
    return is_In/len(max_la_l)

def BLEU_topp(inference, label, n=1):
    max_in_l = []
    max_la_l = []
    for i in range(inference.shape[1]):
        max_in = Process.top_p_decoding(inference[0][i][:]).item()
        max_in_l.append(max_in)
    for i in range(label.shape[1]):
        max_la = torch.argmax(label[0][i][:]).item()
        max_la_l.append(max_la)
    # 计算 n-gram
    ngram_in = [tuple(max_in_l[i:i+n]) for i in range(len(max_in_l)-n+1)]
    ngram_la = [tuple(max_la_l[i:i+n]) for i in range(len(max_la_l)-n+1)]
    # 计算参考文本中的 n-gram 的频次
    count_la = Counter(ngram_la)
    # 计算生成文本中的 n-gram 与参考文本中的 n-gram 的交集
    intersection_count = sum(min(count_la[gram], ngram_in.count(gram)) for gram in set(ngram_in))
    intersection_count = intersection_count if intersection_count != 0 else 1
    ln = len(ngram_in) if len(ngram_in) !=0 else 100000
    # 计算 BLEU 分数+1&1000是为了平滑
    precision = intersection_count / ln
    return precision

def Rouge_1(inference,label):
    max_in_l = []
    max_la_l = []
    for i in range(inference.shape[1]):
        max_in = torch.argmax(inference[0][i][:]).item()
        max_in_l.append(max_in)
    for i in range(label.shape[1]):
        max_la = torch.argmax(label[0][i][:]).item()
        max_la_l.append(max_la)
    is_In = 0
    for i in max_in_l:
        if(i in max_la_l):
            is_In += 1
    if(len(max_la_l) == 0):
        ln = 1000000
    else:
        ln = len(max_la_l)
    return is_In/ln

def BLEU(inference, label, n=1):
    max_in_l = []
    max_la_l = []
    for i in range(inference.shape[1]):
        max_in = torch.argmax(inference[0][i][:]).item()
        max_in_l.append(max_in)
    for i in range(label.shape[1]):
        max_la = torch.argmax(label[0][i][:]).item()
        max_la_l.append(max_la)
    # 计算 n-gram
    ngram_in = [tuple(max_in_l[i:i+n]) for i in range(len(max_in_l)-n+1)]
    ngram_la = [tuple(max_la_l[i:i+n]) for i in range(len(max_la_l)-n+1)]
    # 计算参考文本中的 n-gram 的频次
    count_la = Counter(ngram_la)
    # 计算生成文本中的 n-gram 与参考文本中的 n-gram 的交集
    intersection_count = sum(min(count_la[gram], ngram_in.count(gram)) for gram in set(ngram_in))
    intersection_count = intersection_count if intersection_count != 0 else 1
    ln = len(ngram_in) if len(ngram_in) !=0 else 100000
    # 计算 BLEU 分数+1&1000是为了平滑
    precision = intersection_count / ln
    return precision

def F1(BLEU, rouge_1):
    if BLEU + rouge_1 == 0:
        return 0
    return 2 * BLEU * rouge_1 / (BLEU + rouge_1)