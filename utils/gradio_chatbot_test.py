import time

import gradio as gr
import numpy as np
import os
import random

def response(message, chat_history):
    bot_response = random.choice(['How are you?', 'I love you.', 'I am very hungry.'])
    chat_history.append((message, bot_response))
    time.sleep(2)
    return "", chat_history     # response返回的第一个空值，是为了将输入msg内容置零

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(autofocus=True)
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(response, [msg, chatbot], [msg, chatbot])    # submit对应键盘的回车事件
    demo.launch()