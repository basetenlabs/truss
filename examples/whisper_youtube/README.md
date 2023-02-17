# How to deploy Whisper for YouTube to Baseten

Follow along with the general [README](https://github.com/basetenlabs/truss#truss) and general [deployment guides](https://truss.baseten.co/#deploy-your-model) for details on how to deploy your this model to Baseten.

For this model specifically, import baseten and truss and load + deploy your model using the API key from your Baseten account

(Settings->Account->API keys)

```
import baseten, truss
my_truss = truss.load("PATH TO MODEL") # if in the same directory truss.load("./whisper_youtube")
baseten.login("API_KEY")
baseten.deploy(my_truss)
```

Once the deployment is finished invoke your model with the following input

```
model = baseten.deployed_model_id("MODEL_ID")
model.predict({'url': "YOUTUBE_URL"})
# output is of the shape {'language': <str>, 'text': <str>}
```
