import gradio as gr
import numpy as np


def greet(name, is_morning, temperature):
    salutation = 'Good Morning!' if is_morning else 'Good Afternoon!'
    greeting = f'{salutation} {name}. It is {temperature} degrees today.'
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


def image_sim_classifier(img):
    return {'cat': 0.3, 'dog': 0.7}


# demo = gr.Interface(fn=image_sim_classifier, inputs=gr.Checkbox(value=True, label='同意条款'), outputs='label')

# demo = gr.Interface(fn=lambda text: text[::-1], inputs='text', outputs='text')

# demo = gr.Interface(
#     fn=greet,
#     inputs=['text', 'checkbox', gr.Slider(0, 100, value=17)],
#     outputs=['text', 'number'],
# )
#
# demo.launch()


def flip_text(x):
    return x[::-1]

def flip_image(img):
    return np.fliplr(img)

demo = gr.Blocks()

with demo:
    gr.Markdown("Flip text or image files using this demo")
    with gr.Tab("Flip text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")
    text_button.click(flip_text, text_input, text_output)
    image_button.click(flip_image, image_input, image_output)

if __name__ == '__main__':
    demo.launch()

