FROM baseten/truss-server-base:3.10-gpu-v0.7.16

RUN pip uninstall cython -y

RUN pip install git+https://github.com/coqui-ai/TTS.git
