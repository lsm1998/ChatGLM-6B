from transformers import AutoTokenizer, AutoModel

import torch

# pip3 uninstall torch torchvision torchaudio
# pip3 install -i https://pypi.douban.com/simple torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

print(torch.__version__)
print(torch.version.cuda)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


# 按需修改，目前只支持 4/8 bit 量化
# INT8 量化的模型将"THUDM/chatglm-6b-int4"改为"THUDM/chatglm-6b-int8"
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()

# CPU训练
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()

model = model.eval()

historyList = []

def question(prompt: str):
    response, history = model.chat(tokenizer, prompt, history = historyList)
    print("提问：" , prompt)
    print("回答：" , response)
    historyList.extend(history)


question('你好')
question('请问100乘以20，再加上500等于多少')
question('请问妈妈的爸爸叫什么')