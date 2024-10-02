from transformers import AlbertModel,RobertaModel
import torch.nn as nn
import torch

class Transformer_Classifier(nn.Module):
    def __init__(self,config,Class_num,mode):
        super().__init__()
        if(mode == "albert"):
            self.model = AlbertModel(config, add_pooling_layer=False)
        elif(mode == "Roberta-base-distil"):
            self.model = RobertaModel(config)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.Decode = nn.Linear(config.hidden_size,Class_num)

    def forward(self,input,attn_mask = None):
        out = self.model(input_ids = input,attention_mask = attn_mask)
        out = out.last_hidden_state[:,0,:]
        out = self.norm(out)
        out = self.Decode(out)
        return out
    
    def ClassifierOutput(self,out,mode = "topk",KorP = 2):
        """
        out只接受batch==1
        """
        indexs = None
        weight = None
        out = torch.squeeze(out,dim=0)
        if(mode == "topk"):
            topk,indexs = torch.topk(out,KorP)
            weight = torch.softmax(topk,-1)
        elif(mode == "topp"):
            prob = torch.softmax(out,dim=-1)
            sorted_prob, sorted_index = torch.sort(prob, descending=True)
            cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
            mask = cumulative_prob <= KorP
            mask[..., 1:] = mask[..., :-1].clone()
            #至少保留一个
            mask[..., 0] = 1
            top_p_prob = sorted_prob[mask]
            indexs = sorted_index[mask]
            weight = torch.softmax(top_p_prob,-1)
        return indexs,weight
