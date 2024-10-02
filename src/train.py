import torch
from src import Get_data
from src import Evaluation
from src import Criterion 
from src import Process
from peft import LoraConfig,get_peft_model,PeftModel
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def train(tokenizer,model,lr,Train_datasets,Test_datasets,LoRA_pth,epoch,max,mode,cuda_mem,has_lora = False):  
    print("检查训练模式是否正确，当前载入Lora:",has_lora)
    if(has_lora == True):
        print("尝试获取Lora模型")
        model = PeftModel.from_pretrained(model,LoRA_pth)
        print("获取完成")
    elif(has_lora == False):
        LoraC = LoraConfig(
            target_modules=["q_proj","v_proj"],  # 只对atten.c_proj微调
            r=8,  # LoRA rank
            lora_alpha=16,  # LoRA scaling factor
            lora_dropout=0.1,  # LoRA dropout
            bias="lora_only"  # 不微调bias
        )
        model = get_peft_model(model,LoraC)
    
    for param in model.base_model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(),lr)
    model = model.cuda()
    sm_loss = 0
    o = 0
    sm = 0
    s = 0
    bleu = 0
    rouge = 0
    selfbleu = 0
    rougeL = 0
    selfrouge = 0
    avg_loss_list = []
    for i in range(epoch):
        if(i%2001 == 0 and i != 0):
            model.save_pretrained(f"./finish/lora/{mode}{i-1}step+lr{opt.param_groups[0]['lr']}")
        if(sm == 500):
            s += 1
            avg_loss = sm_loss/o
            avg_loss_list.append(avg_loss)
            print(f"第{s*sm}个epoch,loss:",avg_loss)
            ipt,lb = Get_data.get_QAdataset(Train_datasets)
            print("input:",ipt)
            ipt = [
                {"role": "system", "content": ''},
                {"role": "user", "content": ipt}
            ]
            ipt = tokenizer.apply_chat_template(
                ipt,
                tokenize=True,
                add_generation_prompt=True
            )
            print("out_topp:",Process.generate_text(model,tokenizer,ipt,100,cuda_memory=cuda_mem))
            print("out_gready:",Process.generate_text(model,tokenizer,ipt,100,decoding_strategy='greedy',cuda_memory=cuda_mem))
            print("label:",lb)
            avg_selfrouge = selfrouge/sm
            avg_selfbleu = selfbleu/sm
            avg_rouge = rouge/sm
            avg_rougeL = rougeL/sm
            avg_bleu = bleu/sm
            avg_F1 = Evaluation.F1(avg_selfbleu,avg_selfrouge)
            sm = 0
            sm_loss = 0
            o = 0
            print("bleu:",avg_bleu,"rougeL:",avg_rougeL,"rouge1:",avg_rouge,"avg_selfrouge:",avg_selfrouge,"avg_selfbleu:",avg_selfbleu,"avg_F1:",avg_F1)
            if(Test_datasets!=None):
                Process.test(500,model,Test_datasets,tokenizer,max)
            selfrouge = 0
            selfbleu = 0
            rouge = 0
            bleu = 0
            rougeL = 0
        data = Get_data.get_QAdataset(Train_datasets)
        ipt,lbl,start_idx,lbb = Get_data.deal_inputs_batched_FIXED(tokenizer,data,"",max)
        ipt = torch.tensor([ipt]).cuda()
        lbl = torch.tensor([lbl])
        label = Process.to_one_hot(lbl)
        out = model(ipt)
        out = out.logits
        if(label[:,start_idx+1:,:].shape == torch.Size([1, 0, 151936])):
            continue
        loss = Criterion.AIO("Ce",out[:,start_idx:,:],label[:,start_idx:,:])
        loss = sum(loss)
        o += 1 
        try:
            loss.backward()
            opt.step()
            sm += 1
            sm_loss += loss.item()
            selfbleu += Evaluation.BLEU(out[:,start_idx:,:],label[:,start_idx:,:])
            selfrouge += Evaluation.Rouge_1(out[:,start_idx:,:],label[:,start_idx:,:])
            out = tokenizer.decode(torch.argmax(out[:,start_idx:,:][0],dim=-1))
            bleu += sentence_bleu([lbb],out)
            rg = scorer.score("".join(lbb),out)
            rouge += rg["rouge1"][2]
            rougeL += rg["rougeL"][2] 
        except RuntimeError:
            pass
        opt.zero_grad()
        torch.cuda.empty_cache()
    print(f"总计{i}个epoch,loss:",sm_loss/sm)