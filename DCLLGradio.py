from src import DCLL
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab("Basic_LoRA_Training"):
        with gr.Column():
            lb = gr.HTML("<h3>The primary name we thougt is DCLLFA(DCLL), so in source code some function or variable use these name, but after we worte the paper, we decide change it to DLP-LoRA</p>")
            lbb = gr.HTML("<h3>Notice,whatever what mode you use, you need change src.Process.to_one_hot's vocab_size to your LLMs' vocab_size, and because some LLMs' ending token are not the same you need also change the src.Process.generate_text to your token id(cause it is not the latest one we developed some edit is necessary)</p>")
            Model_input = gr.Textbox(label = "Input your LLMs path,",placeholder="...",value = "./BasicModel/LLMs/Qwen2-1.5B-Instruct")
            isTrain = gr.Checkbox(label="isTrain(if not ticked then is test mode)")
            isTestUseLoRA = gr.Checkbox(label="isUsing LoRA to test")
            Step = gr.Number(label="test/training steps")
            cuda_memory = gr.Number(label="The cuda memory")
            max = gr.Number(label="Context input limitation")
            prompt_text = gr.HTML("<h3>While under LoRA training it will save every 2000 steps,more details in code</p>")
            TrainDatasets = gr.Textbox(label="Input Training path")
            TestDatasets = gr.Textbox(label="Input Testing path",placeholder="<h3>This place is necessary, then choose the mode you would like to use</p><h3>Under Test mode the training datasets and saving name can use any json path you like,it will not infect the feedback</p>")
            LoRA_pth = gr.Textbox(label="LoRA path",placeholder="if under training you can write anything you like")
            modeSave = gr.Textbox(label="Saving name")
            inf = gr.Button("Start！")
            deltatime = gr.Textbox(label="Using time(only testing time,not include Initialize the mode)")
            inf.click(fn = DCLL.ModelWithoutDCLL,inputs=[max,Model_input,isTrain,Step,TrainDatasets,TestDatasets,LoRA_pth,modeSave,cuda_memory,isTestUseLoRA],outputs=[deltatime])
    with gr.Tab("ClassifierTraining"):
        with gr.Column():
            prompt_text_ = gr.HTML("<h3>Classifier training part</p>")
            Model_input = gr.Textbox(label="choose the mode of classifier you like albert/Roberta-base-distil")
            typs = gr.Number(label="The types of traininer")
            prompt_text = gr.HTML("<h3>Training mode ever 2500 save one </p>")
            TrainDatasets = gr.Textbox(label="Input training path")
            TestDatasets = gr.Textbox(label="Input testing path")
            epoch = gr.Number(label="Training epoch")
            batch = gr.Number(label="Training batch")
            inf = gr.Button("Start!")
            isFinish = gr.Textbox()
            inf.click(fn = DCLL.ClassifierTrainer,inputs=[Model_input,typs,TrainDatasets,TestDatasets,epoch,batch],outputs=[isFinish])
    with gr.Tab("Classifier datasets maker Fusion"):
        with gr.Row():
            # 上传文件的输入组件
            file_input_0 = gr.File(label="json", file_types=[".json"])
            out = gr.Textbox()
        inf_ = gr.Button("Start trasforming the json to needed format")
        with gr.Row():
            # 上传文件的输入组件
            file_input_1 = gr.File(label="json1", file_types=[".json"])
            file_input_2 = gr.File(label="json2", file_types=[".json"])
        inf = gr.Button("Start Fusion")
        prompt_text_ = gr.HTML("<h3>The output under ./TestOutputs</p>")
        isFinish = gr.Textbox()
        inf.click(fn = DCLL.jsonReader,inputs=[file_input_1,file_input_2],outputs=[isFinish])
        inf_.click(fn=DCLL.jsonChanger,inputs=[file_input_0],outputs=[out])
    with gr.Tab("DLP-LoRA Test"):
        with gr.Column():
            # 上传文件的输入组件
            lb = gr.HTML("<h3>The primary name we thougt is DCLLFA(DCLL), so in source code some function or variable use these name, but after we worte the paper, we decide change it to DLP-LoRA</p>")
            lbb = gr.HTML("<h3>Notice,whatever what mode you use, you need change src.Process.to_one_hot's vocab_size to your LLMs' vocab_size, and because some LLMs' ending token are not the same you need also change the src.Process.generate_text to your token id(cause it is not the latest one we developed some edit is necessary)</p>")
            file_input = gr.File(label=r"Upload a .txt file include every lora path,\n means the next", file_types=[".txt"])
            prompt_text_ = gr.HTML("<h3>DLP-LoRA</p>")
            mode = gr.Textbox(label="choose mode albert/Roberta-base-distil")
            Model_input = gr.Textbox(label="basic LLM path")
            Classifier_pth = gr.Textbox(label="Classifier model path")
            test_Step = gr.Number(label="test steps")
            mx = gr.Number(label="The context input token limitation num")
            TestDatasets = gr.Textbox(label="testing path")
            inf = gr.Button("Start!")
            deltatime = gr.Textbox()
            inf.click(fn = DCLL.DCLLFA,inputs=[file_input,Model_input,Classifier_pth,TestDatasets,test_Step,mode,mx],outputs=[deltatime])

demo.launch()