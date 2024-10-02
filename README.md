# **DLP-LoRA introduce**

We introduced DLP-LoRA, a dynamic and lightweight plugin that employs a mini-MLP module
with only 5 million parameters to dynamically fuse multiple LoRAs at the sentence level using
top-p sampling strategies. Our comprehensive evaluation across 17 MCQ tasks and 9 QA tasks
demonstrates that DLP-LoRA not only closely matches the performance of individually fine-tuned
single LoRAs but also surpasses them on certain tasks, all while incurring less than twice the in-
ference time. Through detailed discussions and ablation studies, we have shown that DLP-LoRA
effectively balances performance and efficiency in multi-task learning, making it a practical solution
for dynamic multi-task adaptation in LLMs.

arXiv:..................

## **DLP-LoRA quick start**

### 1. LLMs&Classifier Model prepare:

- First you may need to choose the LLM you would like to use, and put them in ./BasicModel/LLMs, and get albert and robert-distil though huggingface.

### 2. DataSets prepare:

- To train your own LoRA or test your DLP-LoRA, you need preprocess your datasets to json which formed by DictList, where each dictionary contains keys "Q" and "A". Then inject them into DataSets.

# DLP-LoRA Gradio Usage

## Basic Page

### 1. Basic LoRA Training

On this page, you can test basic models and train LoRA by filling in the required fields.

**Important Notes**:
- Regardless of the mode you're using, make sure to change the `src.Process.to_one_hot`'s `vocab_size` to match the vocab size of your LLM.
- Since different LLMs have varying end tokens, you also need to update `src.Process.generate_text` with the correct token id for your model’s end token (this part may require editing as it’s not the latest version developed).

**JSON Format**:  
- The format is `DictList`, where each dictionary contains keys "Q" and "A".

**While Training or Testing**: 
- You may need fill the Datasets Path with the write json, and if you are using testing mode of Basic model then LoRA path you can fill anything you like.

**Outputs**
1. ModelOutput
- The model will output to ./finish/lora, where you can find your LoRA model.

2. TestDataOutput:
- The TestData will output to ./TestOutputs, where you can find your test feedback.

### 2. ClassifierTraining

On this page, you can Traing your own Classifier by choose different model of `albert` and `Roberta-base-distil` or `MLP`(MLP config based on the albert so while using MLP choose albert is OK)

**Important Notes**:
- The datasets which training for LoRA can be fully used by Classifier, you can use Page 3`Classifier datasets maker Fusion` to invert your LoRA datasets to Classifier datasets.

- Notice only the right format formed by Page 3 could be used to train the classifier, more details under the code

**While Training or Testing**: 
- You may need fill the Datasets Path with the write json, and if you are using testing mode of Basic model then LoRA path you can fill anything you like.

**Outputs**
1. ModelOutput
- The model will output to ./finish/Classifier, where you can find your Classifier model

### 3. Classifier datasets maker Fusion

On this page, you can invert your LoRA datasets to Classifier datasets.

**How to use**:
1. The datasets which training for LoRA can be fully used by Classifier, you can just put your LoRA datasets which keep same format of page 1 then push the start it will out put to your folder.

2. Cause you are training the classifier so we designed the fusion function to help you fusion your different datasets

- Notice is better to sample same number of the datasets, then may give a better feed back.

### 4. DLP-LoRA Test

While you got the LoRA and Classifier, is time to start DLP-LoRA

**How to use**:
1. Firstly, you need form a `LoRA pathList`.txt and push it into the blank for our model to find your LoRAs.

2. Then you need choose the type of your classifier and filling other path follow the page's order.

3. While you filling all blank, choose start then it will start testing your DLP-LoRA

**JSON Format**:  
- The format is `DictList`, where each dictionary contains keys "Q" and "A".

**Outputs**
2. TestDataOutput:
- The TestData will output to ./TestOutputs, where you can find your DLP-LoRA test feed back.

**Notice the empty folder will not show in Github**
```
├── BasicModel
│   ├── Classifiers
│   └── LLMs
├── DataSets
├── DCLLGradio.py
├── finish
│   ├── Classifier
│   └── lora
├── LoRAdicts
├── LoRA_models
│   └── formList.py
├── model
├── PrintDict.py
├── ReadMe.md
├── src
│   ├── ClassifierUtilities.py
│   ├── Criterion.py
│   ├── DCLL.py
│   ├── Evaluation.py
│   ├── Get_data.py
│   ├── LoraModel.py
│   ├── Process.py
│   ├── train.py
│   ├── __init__.py
│   └── __pycache__
│       ├── Add_mask.cpython-310.pyc
│       ├── Add_mask.cpython-38.pyc
│       ├── Add_mask.cpython-39.pyc
│       ├── Berts.cpython-310.pyc
│       ├── Berts.cpython-38.pyc
│       ├── Berts.cpython-39.pyc
│       ├── ClassifierUtilities.cpython-38.pyc
│       ├── Criterion.cpython-310.pyc
│       ├── Criterion.cpython-38.pyc
│       ├── Criterion.cpython-39.pyc
│       ├── DCLL.cpython-311.pyc
│       ├── DCLL.cpython-38.pyc
│       ├── Evaluation.cpython-310.pyc
│       ├── Evaluation.cpython-38.pyc
│       ├── Evaluation.cpython-39.pyc
│       ├── Get_data.cpython-310.pyc
│       ├── Get_data.cpython-38.pyc
│       ├── Get_data.cpython-39.pyc
│       ├── LoraModel.cpython-38.pyc
│       ├── PictureDrawer.cpython-38.pyc
│       ├── Process.cpython-38.pyc
│       ├── Processing.cpython-310.pyc
│       ├── Processing.cpython-38.pyc
│       ├── Processing.cpython-39.pyc
│       ├── train.cpython-311.pyc
│       ├── train.cpython-38.pyc
│       ├── train.cpython-39.pyc
│       ├── Train_mode.cpython-310.pyc
│       ├── Train_mode.cpython-38.pyc
│       ├── Train_mode.cpython-39.pyc
│       ├── Transform_parameters.cpython-310.pyc
│       ├── Transform_parameters.cpython-38.pyc
│       ├── Transform_parameters.cpython-39.pyc
│       ├── __init__.cpython-310.pyc
│       ├── __init__.cpython-311.pyc
│       ├── __init__.cpython-38.pyc
│       └── __init__.cpython-39.pyc
├── srcClassifier
│   ├── ClassifierDatasetsMaker.py
│   ├── ClassifierModel.py
│   ├── Criterion.py
│   ├── evaluation.py
│   ├── GetDatasets.py
│   ├── Get_data.py
│   ├── Process.py
│   ├── Trainer.py
│   ├── __init__.py
│   └── __pycache__
│       ├── Add_mask.cpython-310.pyc
│       ├── Add_mask.cpython-38.pyc
│       ├── Add_mask.cpython-39.pyc
│       ├── Berts.cpython-310.pyc
│       ├── Berts.cpython-38.pyc
│       ├── Berts.cpython-39.pyc
│       ├── ClassifierModel.cpython-311.pyc
│       ├── ClassifierModel.cpython-38.pyc
│       ├── Criterion.cpython-310.pyc
│       ├── Criterion.cpython-38.pyc
│       ├── Criterion.cpython-39.pyc
│       ├── Evaluation.cpython-310.pyc
│       ├── evaluation.cpython-38.pyc
│       ├── Evaluation.cpython-39.pyc
│       ├── GetDatasets.cpython-38.pyc
│       ├── Get_data.cpython-310.pyc
│       ├── Get_data.cpython-38.pyc
│       ├── Get_data.cpython-39.pyc
│       ├── LoraModel.cpython-38.pyc
│       ├── Process.cpython-38.pyc
│       ├── Processing.cpython-310.pyc
│       ├── Processing.cpython-38.pyc
│       ├── Processing.cpython-39.pyc
│       ├── train.cpython-311.pyc
│       ├── train.cpython-38.pyc
│       ├── train.cpython-39.pyc
│       ├── Trainer.cpython-38.pyc
│       ├── Train_mode.cpython-310.pyc
│       ├── Train_mode.cpython-38.pyc
│       ├── Train_mode.cpython-39.pyc
│       ├── Transform_parameters.cpython-310.pyc
│       ├── Transform_parameters.cpython-38.pyc
│       ├── Transform_parameters.cpython-39.pyc
│       ├── __init__.cpython-310.pyc
│       ├── __init__.cpython-311.pyc
│       ├── __init__.cpython-38.pyc
│       └── __init__.cpython-39.pyc
├── startGradio.bat
├── TestOutputs
│   └── Fusion
└── __pycache__
    ├── DCLL.cpython-311.pyc
    └── DCLL.cpython-38.pyc```
