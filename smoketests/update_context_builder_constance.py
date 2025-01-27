import json

import click
import requests
import utils

from truss.remote import baseten

_KEY = "ORGANIZATION_TRUSS_CONTEXT_BUILDER_OVERRIDES"


@click.command()
@click.option("--version", required=True, help="Truss version as on pypi.")
def main(version: str) -> None:
    # Get user from api key.
    remote = baseten.BasetenRemote(utils.BASETEN_REMOTE_URL, utils.BASETEN_API_KEY)
    user = remote.whoami()
    user_name = user.user_email.replace(".", "--").replace("+", "--").replace("@", "--")

    api_url = f"{utils.BASETEN_REMOTE_URL}/constance/settings/{_KEY}"
    headers = {"Authorization": f"Api-Key {utils.BASETEN_API_KEY}"}

    # Get current overrides.
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    data = response.json()
    current_overrides = json.loads(data["value"])

    # Update to new version.
    current_overrides[user_name] = f"baseten/truss-context-builder:v{version}"
    payload = {"value": json.dumps(current_overrides, indent=4)}
    response = requests.post(api_url, headers=headers, data=payload)
    response.raise_for_status()
    if error := response.json().get("error", None):
        click.echo(error)
    else:
        click.echo(
            f"Successfully updated context builder for `{user_name}` "
            f"to `{current_overrides[user_name]}`."
        )


if __name__ == "__main__":
    main()
