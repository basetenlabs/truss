
import gradio as gr

def output_transform(d):return d.data

def run_model(prompt,temperature):
    import truss
    th = truss.load("local_dev/test_truss")
    out = th.predict({'prompt':prompt,'temperature':temperature})
    print(out)
    return output_transform(out)

demo = gr.Interface(fn=run_model, inputs=['text', 'number'], outputs="image")
demo.launch()

