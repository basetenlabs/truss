from flask import Flask
import os
import signal
import subprocess

app = Flask(__name__)


INFERENCE_SERVER_PROCESS = None


@app.route("/patch")
def patch():
    return "<p>Hello, World!</p>"


@app.route("/restart_inference_server")
def restart_inference_server():
    inference_app_home = os.environ['APP_HOME']
    global INFERENCE_SERVER_PROCESS
    if INFERENCE_SERVER_PROCESS is not None:
        try:
            INFERENCE_SERVER_PROCESS.kill()
        except: # todo something
            pass
    import ipdb;ipdb.set_trace()
    cwd = os.getcwd()
    os.chdir(inference_app_home)
    try:
        INFERENCE_SERVER_PROCESS = subprocess.Popen(
            ['/usr/local/bin/python', f'{inference_app_home}/server/inference_server.py'],
        )
    finally:
        os.chdir(cwd)

    return {'msg': 'Inference server started successfully'}


@app.route("/stop_inference_server")
def stop_inference_server():
    # todo
    return {'msg': 'Inference server stopped successfully'}

if __name__ == "__main__":
    from waitress import serve
    print('Starting control server')
    serve(app, host="0.0.0.0", port=8090)