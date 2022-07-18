# GPU Support

GPU support is a newer feature and limited to certain hardware, but we use it in production at Baseten.

GPU support is independent from what hardware your model was trained on. Rather, it asks if your model requires a GPU to run predictions.

Essentially, to enable GPU support, just set `gpu: true` in the config file and the Docker image updates to include CUDA. This Docker image will run on many, but not all, devices that don't support CUDA with a substantial reduction in performance.

Non-CUDA GPUs are not yet supported.
