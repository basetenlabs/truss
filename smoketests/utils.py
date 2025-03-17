import os

BACKEND_ENV_DOMAIN = "staging.baseten.co"
BASETEN_API_KEY = os.environ["BASETEN_API_KEY_STAGING"]
BASETEN_REMOTE_URL = f"https://app.{BACKEND_ENV_DOMAIN}"
