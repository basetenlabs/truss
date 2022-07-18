# GPU support

{% hint style="warning" %}

GPU support is a newer feature and limited to certain hardware, but we use it in production at Baseten.

{% endhint %}

Large machine learning models are generally trained on a CUDA-based GPU. Once trained, some can return predictions on a standard CPU, but others require access to GPU hardware. For these cases, Truss creates a model serving environment that gives GPU access.

To enable GPU support, go to your Truss' `config.yaml` and set:

```
resources:
  use_gpu: true
```

This will ensure the Docker image created to serve your Truss includes CUDA and accesses the system's GPU, if available. This Docker image will run on many, but not all, devices that don't support CUDA with a substantial reduction in performance.
