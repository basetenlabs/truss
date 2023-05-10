# AI Search Example

This demonstrates how to build an AI Search engine using truss

## Test Locally

```
In [1]: import truss
In [2]: tr = truss.load("./examples/ai-search/")
In [3]: tr.predict({"query": "animal", "limit":1})
Out[3]:
[{'payload': {'text': 'The quick brown fox jumps over the lazy dog.'},
  'score': 0.3724476099014282}]
```


## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
import baseten, truss. os

tr = truss.load("./examples/ai-search/")

baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(tr, model_name"AI Search", publish=True)
```
