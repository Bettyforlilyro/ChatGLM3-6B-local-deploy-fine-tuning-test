import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from typing import Dict
from transformers import AutoTokenizer, AutoModel
import config.data_config as data_config

model_dir = data_config.model_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

app = FastAPI()
@app.post('/chat')
def chat(data: Dict):
    query = data['query']
    history = data['history']
    if history == "":
        history = []
    response, history = model.chat(tokenizer, query, history=history, top_p=0.95, temperature=0.95)
    response = {"response": response, "history": history}
    return JSONResponse(response)


if __name__ == '__main__':
    uvicorn.run(app, host=data_config.host_ip, port=int(data_config.host_port))
    # 控制台代码
    # import requests
    # import json
    #
    # data = {"query": "你好", "history": ""}
    # json_data = json.dumps(data)
    # response = requests.post('http://' + data_config.host_ip + ':' + str(data_config.host_port) + '/chat', data=json_data).json()
    # print(response)
    # response_chat = response['response']
    # history = response['history']
