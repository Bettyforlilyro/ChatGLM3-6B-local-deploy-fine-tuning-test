import os
import numpy as np
import gradio as gr
from zipfile import ZipFile

def zip_to_json(file_obj):
    files = []
    with ZipFile(file_obj) as zf:
        for zip_info in zf.infolist():
            files.append({
                'filename': zip_info.filename,
                'size': zip_info.file_size,
                'compress_size': zip_info.compress_size
            })
    return files

demo = gr.Interface(fn=zip_to_json, inputs='file', outputs='json')

if __name__ == '__main__':
    demo.launch()