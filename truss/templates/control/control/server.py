from flask import Flask

app = Flask(__name__)

@app.route("/patch")
def patch():
    return "<p>Hello, World!</p>"


@app.route("/restart_inference_server")
def restart_inference_server():
    # todo
    return {'msg': 'Inference server started successfully'}


@app.route("/stop_inference_server")
def stop_inference_server():
    # todo
    return {'msg': 'Inference server stopped successfully'}

if __name__ == "__main__":
    from waitress import serve
    print('Starting control server')
    serve(app, host="0.0.0.0", port=8090)