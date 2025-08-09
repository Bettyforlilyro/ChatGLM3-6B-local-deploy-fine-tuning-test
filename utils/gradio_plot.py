import time

import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

def plot_line(style, progress=gr.Progress()):
    progress(0, "Loading data...")
    time.sleep(0.5)
    for i in progress.tqdm(range(100)):
        time.sleep(0.03)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(2025, 2040 + 1)
    year_count = x.shape[0]
    plt_format = ({'cross': "X", 'line': "-", "circle": 'o--'})[style]
    series = np.arange(0, year_count, dtype=np.float32)
    series = series**2
    series += np.random.rand(year_count)
    ax.plot(x, series, plt_format)
    return fig

app = gr.Interface(fn=plot_line, inputs=gr.Dropdown(["cross", "line", "circle"]), outputs=gr.Plot(label="picc"))
app.launch()