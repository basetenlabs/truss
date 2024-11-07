# Model how-to

Welcome to your custom model's Truss. Below are some useful commands to work with this Truss. For all the commands below, make sure you're in this directory when you run them. You can find the docs for Truss [here](https://truss.baseten.co).

### Build a Docker image from your Truss
```
truss build-image
```

### Run the Docker image from your Truss
```
# We assume you've built the image first.
truss run-image
```

### Run inference on your Truss
There are two ways to run inference on your Truss model.
#### Via Truss CLI
```
truss predict --target_directory ./  --request 'YOUR_INPUT'

```

#### Via CURL
In order to run inference via CURL, we assume you've built and run the Docker image generated from your Truss. Refer above for more instructions on how to do this.
```
curl -H 'Content-Type: application/json' \
-d 'YOUR_INPUT' \
-X POST http://localhost:8080/v1/models/model:predict


```

### Running all of the example inputs on your Truss
```
truss run-example
```

### Run a specific example input on your Truss
```
truss run-example --name EXAMPLE_NAME

```

### Adding pre/postprocessing code
You may find that you'd like to preprocess the inputs to your model or postprocess the outputs of your model.
1. Navigate to `model/model.py`. This is where your preprocessing/postprocessing logic will live.
2. Define a `preprocess` or `postprocess` function as such.
```
def preprocess(self, request: Dict) -> Dict:
    # Code that runs before inference
    return request

def postprocess(self, request: Dict) -> Dict:
    # Code that runs after inference
    return request
```
