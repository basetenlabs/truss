## Flan-T5 XL Truss

This is a [Truss](https://truss.baseten.co/) for Flan-T5 XL using `T5Tokenizer` and `T5ForConditionalGeneration` from the `transformers` library. This README will walk you through how to deploy this Truss on [Baseten](https://www.baseten.co/) to get your own instance of a Flan-T5 XL model.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten and other platforms. Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy models on Baseten.

## Deploying Flan-T5 XL

To deploy the Flan-T5 XL Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the Flan-T5 XL Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss
flan_t5_xl_truss = truss.load("path/to/flan-t5-xl")
```

4. (optional) __Set your Hugging Face User Access Token__: If you want to pass in a Hugging Face user access token, make sure to [create a secret in your Baseten account](https://docs.baseten.co/settings/secrets) named `hf_access_token` with the value of your user access token. Read more about secret management in Truss [here](https://truss.baseten.co/develop/secrets).

5. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

6. __Deploy the Flan-T5 XL Truss__: Deploy Flan-T5 XL to Baseten with the following command:
```
# Include is_trusted=True if you're using a secret defined in Baseten, otherwise it can be removed
baseten.deploy(flan_t5_xl_truss, is_trusted=True)
```

Once your Truss is deployed, you can start using Flan-T5 XL through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Flan-T5 XL API Documentation
This section provides an overview of the Flan-T5 XL API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate images based on the provided parameters.

### API Route: `predict`
The predict route is the primary method for generating images based on a given set of parameters. It takes several parameters, only one of which is required:

- __prompt__: (required) This can contain anything, such as written instructions for the model to execute or a problem you'd like solved.

The API also supports passing any additional parameter listed [here](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).

#### Example Usage
You can use the `baseten` model package to invoke your model from Python
```
import baseten
# You can retrieve your deployed model version ID from the UI
model = baseten.deployed_model_version_id('YOUR_MODEL_VERSION_ID')

request = {
    "prompt": "What is 1+1? Explain your reasoning"
}

response = model.predict(request)
```

You can also invoke your model via a REST API
```
curl -X POST "https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt" : "What is 1+1? Explain your reasoning"
         }'
```
