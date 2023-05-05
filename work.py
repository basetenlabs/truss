import truss
th = truss.load('local_dev/test_truss')
print(th.generate_gradio_ui())
