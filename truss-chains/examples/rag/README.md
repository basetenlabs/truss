# Chains RAG example

Retrieval-augmented generation (RAG) is a multi-model pipeline for generating context-aware answers from LLMs. This example implements a basic RAG pipeline with as a Chain.

For a detailed guide to using this example, see the [RAG example in the docs](https://docs.baseten.co/chains/examples/build-rag).

## Development setup

Clone the repository and `cd` into this folder:

```sh
git clone https://github.com/basetenlabs/truss/
cd truss/truss-chains/examples/rag
```

Install dependencies:

```sh
pip install --upgrade 'truss>=0.9.16' 'pydantic>=2.0.0' chromadb
```

Deploy the Phi model:

1. Go to the [Baseten model library](https://www.baseten.co/library/phi-3-mini-4k-instruct/)
2. Deploy the model
3. Copy your deployed model URL (formatted like `https://model-<model_id>.api.baseten.co/production/predict`)
4. Create an API key and save it with `export BASETEN_API_KEY="your_api_key"`

You're ready to go!

## Usage

Test the Chain locally:

```sh
python rag_chain.py
```

Deploy the Chain to production:

```sh
truss chains deploy rag_chain.py
```

Call the deployed chain:

```sh
# You must update the missing values in the Python file first
python call_rag.py
```

## Learn more

For a detailed guide to using this example, see the [RAG example in the docs](https://docs.baseten.co/chains/examples/build-rag).
