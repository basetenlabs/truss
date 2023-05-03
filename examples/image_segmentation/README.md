# Image Segmentation pytorch example.

This is an example of a pytorch model

## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
import baseten, truss. os

sd = truss.load("./examples/stable-diffusion-2-1")

baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(sd, model_name"My Pytorch Model", publish=True)
```
