import torch
import peft
import copy
import torch.nn as nn
import types
from transformers import AutoModelForCausalLM
from peft.tuners.lora.layer import Linear as LoRALinear

def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                result = result + self.MutiLoRAFusion.speedup(dropout(x),self.lyrs,self.typa) * scaling

            result = result.to(torch_result_dtype)
        return result

def forwardForModuleList(self,x):
    out = x.to(self[0].weight.dtype)
    for i in self:
        out = i(out)
    return out


class MutiLoRA:
    def __init__(self,pth_list,LLMs_pth):
        model = AutoModelForCausalLM.from_pretrained(
            LLMs_pth,
            torch_dtype="auto",
            device_map="cuda:0",
        )
        self.model = model
        self.peft = copy.copy(model)
        self.pth_list = pth_list
        self.q = None
        self.v = None
        self.ids = None
        self.score = None
        self.speedup = None

    def get_LoraParam_OneLayer(self,typ,is_bias = False,is_train = False):
        loras = []
        for i in self.pth_list:
            dics = nn.ModuleList([])
            j = 0
            #print(self.model)
            self.peft = peft.PeftModel.from_pretrained(self.model,i)
            #print(self.model)
            dcs = []
            for name, param in self.peft.named_parameters():
                j_ = j
                if "lora_A" in name and typ in name:
                    dic = param
                    a = nn.Linear(dic.shape[-1],dic.shape[0],is_bias)
                    a.weight = dic
                    if(is_train == True):
                        a.requires_grad = True
                if "lora_B" in name and typ in name:
                    dic = param
                    b = nn.Linear(dic.shape[-1],dic.shape[0],is_bias)
                    b.weight = dic
                    if(is_train == True):
                        b.requires_grad = True
                    j+=1
                if j != j_:
                    dcs = nn.ModuleList([a,b])
                    dcs.forward = types.MethodType(forwardForModuleList, dcs)
                    dcs.cuda()
                    dics.append(dcs)
            loras.append(dics)
        return loras

    def collect_LoraParam(self):
        print("尝试获取Lora模型")
        self.q = self.get_LoraParam_OneLayer("q_proj",is_bias=False)
        self.v = self.get_LoraParam_OneLayer("v_proj",is_bias=False)
        self.speedup = ParallelSpdUp(self.q,self.v)
        del self.peft
        del self.model
        torch.cuda.empty_cache()
        print("获取完成")

    def init_ForOneTerm(self,ids,score):
        if(self.q == None or self.v == None):
            self.collect_LoraParam()
        self.ids = ids
        self.score = score
        self.speedup.init_Linears(ids,score)

class ParallelSpdUp(nn.Module):
    def __init__(self,lora_q,lora_v):
        super().__init__()
        self.q = lora_q
        self.v = lora_v
        self.weight = None
        self.ids = None

    def init_Linears(self,ids,weight):
        if(self.weight != None or self.ids != None):
            self.weight = None
            self.ids = None
        self.weight = weight
        self.ids = ids
        
    def forward(self, x, layer, typ):
        if self.ids is None:
            raise RuntimeError("Linear layers have not been initialized. Call init_Linears first.")
        torch.cuda.empty_cache()
        out = []
        for i in self.ids:
            if typ == "q":
                moduleQ = self.q[i][layer]
                out.append(moduleQ(x))
            elif typ == "v":
                moduleV = self.v[i][layer]
                out.append(moduleV(x))
            else:
                raise ValueError("Invalid type. Expected 'q' or 'v'.")
        output = 0
        for n,m in enumerate(self.weight):
            output += out[n]*m
        return output

        
"""
class ParallelLinearLayers(nn.Module):
    def __init__(self, layer_dims):
        super(ParallelLinearLayers, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features) 
            for in_features, out_features in layer_dims
        ])

    def forward(self, x):
        outputs = [layer(x) for layer in self.layers]
        return torch.cat(outputs, dim=-1)
CodeWriter:YuXuan Zhang
"""


class CausalFusionLM(nn.Module):
    def __init__(self,PeftModel,DCLLFA):
        super().__init__()
        self.Model = PeftModel
        self.MutiLoRAFusion = DCLLFA
        #初始化更换函数
        #此处导致参数产生偏差
        self.preprocess(layer_num=28)

    def preprocess(self,layer_num = 28):
        for i in range(layer_num):
            self.Model.base_model.model.model.layers[i].self_attn.q_proj.forward = types.MethodType(forward, self.Model.base_model.model.model.layers[i].self_attn.q_proj)
            self.Model.base_model.model.model.layers[i].self_attn.q_proj.MutiLoRAFusion = self.MutiLoRAFusion
            self.Model.base_model.model.model.layers[i].self_attn.q_proj.lyrs = i
            self.Model.base_model.model.model.layers[i].self_attn.q_proj.typa = "q"
            self.Model.base_model.model.model.layers[i].self_attn.v_proj.forward = types.MethodType(forward, self.Model.base_model.model.model.layers[i].self_attn.v_proj)
            self.Model.base_model.model.model.layers[i].self_attn.v_proj.MutiLoRAFusion = self.MutiLoRAFusion 
            self.Model.base_model.model.model.layers[i].self_attn.v_proj.lyrs = i
            self.Model.base_model.model.model.layers[i].self_attn.v_proj.typa = "v"
    
    def forward(self,x):
        out = self.Model(x)
        return out