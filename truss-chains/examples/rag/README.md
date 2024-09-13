# Chains RAG example

Retrieval-augmented generation (RAG) is a multi-model pipeline for generating
context-aware answers from LLMs. This example implements a basic RAG pipeline
with as a Chain.

For a detailed guide to using this example, see
the [RAG example in the docs](https://docs.baseten.co/chains/examples/build-rag).

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

1. Go to the
   [Baseten model library](https://www.baseten.co/library/phi-3-mini-4k-instruct/)
   and deploy the Phi-3 model.
3. Copy your deployed model URL (formatted like
   `https://model-<MODEL_ID>.api.baseten.co/production/predict`) and update
    `LLM_PREDICT_URL` in `rag_chain.py` with *your* URL.
4. Create a Baseten API key and make it available in your terminal it with
   `export BASETEN_API_KEY="your_api_key"` (either in each session or adding it
   to `.bashrc`).

You're ready to go!

## Usage

Test the Chain locally:

```sh
python rag_chain.py
```

Deploy the Chain to production:

```sh
truss chains push rag_chain.py
```

Note that this command will print you with an example cURL command how to
call it, you only need to update the JSON payload.

For example a chain invocation might look like this (you need to update the
URL):

```sh
curl -X POST 'https://chain-<CHAIN_ID>.api.baseten.co/development/run_remote' \
    -H "Authorization: Api-Key $BASETEN_API_KEY" \
    -d '{"new_bio": "Sam just moved to Manhattan for his new job at a large bank.In college, he enjoyed building sets for student plays."}'
```

## Learn more

For a detailed guide to using this example, see
the [RAG example in the docs](https://docs.baseten.co/chains/examples/build-rag).
