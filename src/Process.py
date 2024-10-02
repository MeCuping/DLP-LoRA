import os
import time
import json
import torch
from src import Get_data, Evaluation
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from src import ClassifierUtilities

def to_one_hot(toked, batch_size=1, vocab_size=151936, is_cuda=True):
    tokens = torch.zeros([toked.size(0), toked.shape[-1], vocab_size])
    for i in range(batch_size):
        for m, n in enumerate(toked[i]):
            tokens[i][m][n] = 1
    if is_cuda:
        return tokens.cuda()
    else:
        return tokens

def t_softmax(logits, temperature=1.0):
    tem = logits / temperature
    prob = torch.softmax(tem, dim=-1)
    return prob

def test_out(test,tokenizer):
    st = []
    for l in range(test.shape[0]):
        for p in range(test.shape[1]):
            st.append(torch.argmax(test[l][p][:]))
    return tokenizer.decode(st)

def top_p_decoding(logits, temperature=1.0, p=0.7):
    prob = t_softmax(logits, temperature)
    sorted_prob, sorted_index = torch.sort(prob, descending=True)
    cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
    mask = cumulative_prob <= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 1
    top_p_prob = sorted_prob[mask]
    top_p_index = sorted_index[mask]
    top_p_prob /= top_p_prob.sum()
    next_token = torch.multinomial(top_p_prob, num_samples=1)
    return top_p_index[next_token]

def greedy_decoding(logits):
    return torch.argmax(logits, dim=-1)

def generate_text(model, tokenizer, input, max_length=100, temperature=1.0, p=0.7, decoding_strategy='top_p',cuda_memory = 19):
    model.eval()
    len_i = len(input)
    max_cuda = 1024*1024*1024*cuda_memory
    with torch.no_grad():
        for _ in range(max_length):
            cuda = "cuda:0"
            torch.cuda.empty_cache()
            if(torch.cuda.memory_allocated(cuda) >= max_cuda):
                break
            de = torch.tensor([input]).cuda()
            out = model(de)
            out = out.logits
            if decoding_strategy == 'top_p':
                out = top_p_decoding(out[:, -1, :], temperature, p)
            elif decoding_strategy == 'greedy':
                out = greedy_decoding(out[:, -1, :])
            out = out[0].tolist()
            input.append(out)
            if input[-1] == 151645:
                break
    
    input = tokenizer.decode(input[len_i:-1])
    model.train()
    return input

def get_next_filename(directory = "./TestOutputs"):
    files = os.listdir(directory)
    json_files = [f for f in files if f.endswith('.json')]
    
    max_number = -1
    for file in json_files:
        try:
            number = int(file.split('.')[0])  
            if number > max_number:
                max_number = number
        except ValueError:
            pass  
    next_number = max_number + 1
    return os.path.join(directory, f"{next_number}.json"),next_number

def test(test_step,model,datasets,tokenizer,max,mode = None):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    selfbleu = 0
    selfrouge = 0
    rougeL = 0
    bleu = 0
    rouge = 0
    sm = 0
    tim1 = time.time()
    for i in range(test_step):
        data = Get_data.get_QAdataset(datasets)
        inp,lb,start_idx,lbb = Get_data.deal_inputs_batched_FIXED(tokenizer,data,"",max)
        label = to_one_hot(torch.tensor([lb]))
        inp = torch.tensor([inp]).cuda()
        out = model(inp)
        out = out.logits
        selfbleu += Evaluation.BLEU(out[:,start_idx:,:],label[:,start_idx:,:])
        selfrouge += Evaluation.Rouge_1(out[:,start_idx:,:],label[:,start_idx:,:])
        out = tokenizer.decode(torch.argmax(out[:,start_idx:,:][0],dim=-1))
        bleu += sentence_bleu([lbb],out)
        rg = scorer.score("".join(lbb),out)
        rouge += rg["rouge1"][2]
        rougeL += rg["rougeL"][2] 
        sm += 1
    tim2 = time.time()
    print("time:",tim2-tim1)
    print("Tst,SelfBleu:",selfbleu/sm,"SelfRouge:",selfrouge/sm,"Bleu:",bleu/sm,"Rouge1:",rouge/sm,"RougeL:",rougeL/sm)
    model.train()
    if(mode != None):
        testData_Out = [{"TestType":mode,"TestStep":test_step,"time":tim2-tim1,"SelfBleu":selfbleu/sm,"SelfRouge":selfrouge/sm,"Bleu":bleu/sm,"Rouge1":rouge/sm,"RougeL":rougeL/sm}]
        filename,__ = get_next_filename()
        with open(filename, 'w') as f:
            json.dump(testData_Out, f, indent=4)
        print(f"Data saved to {filename}")
    return tim2-tim1

def testMultiAdapter(Classifier,test_step,model,datasets,tokenizer,ClassifierTokenizer,max):
    Classifier.eval()
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    selfbleu = 0
    selfrouge = 0
    rougeL = 0
    bleu = 0
    rouge = 0
    sm = 0
    tim1 = time.time()
    for i in range(test_step):
        data = Get_data.get_QAdataset(datasets)
        inp,lb,start_idx,lbb = Get_data.deal_inputs_batched_FIXED(tokenizer,data,"",max)
        loras,weight = ClassifierUtilities.ClassifierInf(data[0],Classifier,ClassifierTokenizer,"topp",max,0.6)
        model.MutiLoRAFusion.init_ForOneTerm(loras,weight)
        label = to_one_hot(torch.tensor([lb]))
        inp = torch.tensor([inp]).cuda()
        out = model(inp)
        out = out.logits
        selfbleu += Evaluation.BLEU(out[:,start_idx:,:],label[:,start_idx:,:])
        selfrouge += Evaluation.Rouge_1(out[:,start_idx:,:],label[:,start_idx:,:])
        out = tokenizer.decode(torch.argmax(out[:,start_idx:,:][0],dim=-1))
        bleu += sentence_bleu([lbb],out)
        rg = scorer.score("".join(lbb),out)
        rouge += rg["rouge1"][2]
        rougeL += rg["rougeL"][2] 
        sm += 1
    tim2 = time.time()
    print("time:",tim2-tim1)
    print("Tst,SelfBleu:",selfbleu/sm,"SelfRouge:",selfrouge/sm,"Bleu:",bleu/sm,"Rouge1:",rouge/sm,"RougeL:",rougeL/sm)
    model.train()
    filename,__ = get_next_filename("./TestOutputs/DCLLFA")
    testData_Out = [{"TestType":f"DCLLFA{__}","TestStep":test_step,"time":tim2-tim1,"SelfBleu":selfbleu/sm,"SelfRouge":selfrouge/sm,"Bleu":bleu/sm,"Rouge1":rouge/sm,"RougeL":rougeL/sm}]
    with open(filename, 'w') as f:
        json.dump(testData_Out, f, indent=4)
    print(f"Data saved to {filename}")
    return tim2-tim1

def generate_text_(model, tokenizer, input, max_length=256, pad_num=1, temperature=1.0, p=0.7, decoding_strategy='top_p'):
    """
    这个将不会整合文字,以列表的形式输出
    """
    de = input+" A:[CLS]"
    model = model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            de = tokenizer.tokenize(de)
            de_ids = tokenizer.encode(de, add_special_tokens=False)
            de_tensor = torch.tensor([de_ids]).cuda()
            out = model(de_tensor[:, :512])
            out = out.logits
            if decoding_strategy == 'top_p':
                out = top_p_decoding(out[:, -1, :], temperature, p)
            elif decoding_strategy == 'greedy':
                out = greedy_decoding(out[:, -1, :])
            de.append(tokenizer.decode(out))
            de = "".join(de).replace("[MASK]", "", 1)
            if tokenizer.decode(out) == tokenizer.sep_token:
                break
    
    de = de.replace("[MASK]", "")
    return tokenizer.tokenize(de)

def read_json_files_from_folder(folder_path):
    combined_data = []

    # 遍历文件夹中的每个文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 读取 JSON 文件
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    if isinstance(data, list):  # 确保读取的内容是数组
                        combined_data.extend(data)
                    else:
                        print(f"Warning: {file_name} does not contain a JSON array.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return combined_data

def GetAllDatasetDir(folder_path):
    Train = []
    test = []
    Name = []
    for file_name in os.listdir(folder_path):
        pathRoot = os.path.join(folder_path,file_name)
        if os.path.isdir(pathRoot):
            Name.append(file_name)
            for j in os.listdir(pathRoot):
                if j == "Train.json":
                    pathDataset = os.path.join(pathRoot,j)
                    Train.append(pathDataset)
                if j == "test.json":
                    pathDataset = os.path.join(pathRoot,j)
                    test.append(pathDataset)
    return Train,test,Name

def GetAllLoRAsDir(folder_path):
    LoRAs = []
    for file_name in os.listdir(folder_path):
        pathRoot = os.path.join(folder_path,file_name)
        if os.path.isdir(pathRoot):
            LoRAs.append(pathRoot)
    return LoRAs

def save_combined_data_to_file(data, output_file_path = r".\TestOutputs\combined_data.json"):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)
        print(f"Combined data saved to {output_file_path}")