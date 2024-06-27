import os

import requests

# Insert the predict URL from the deployed rag chain. You can get it from the CLI
# output or the status page, e.g.
# "https://model-6wgeygoq.api.baseten.co/production/predict".
RAG_CHAIN_URL = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]

new_bio = """
Sam just moved to Manhattan for his new job at a large bank.
In college, he enjoyed building sets for student plays.
"""

if not RAG_CHAIN_URL:
    raise ValueError("Please insert the predict URL for the RAG chain.")

resp = requests.post(
    RAG_CHAIN_URL,
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={"new_bio": new_bio},
)

print(resp.json())
