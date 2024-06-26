import os

import requests

model_id = ""  # fill in the model ID from the API endpoint here
environment = "development"  # development or production
baseten_api_key = os.environ["BASETEN_API_KEY"]

new_bio = """
Sam just moved to Manhattan for his new job at a large bank.
In college, he enjoyed building sets for student plays.
"""

resp = requests.post(
    f"https://model-{model_id}.api.baseten.co/{environment}/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={"new_bio": new_bio},
)

print(resp.json())
