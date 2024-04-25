import logging

logging.basicConfig(level=logging.WARNING)


import click
import truss_chains as chains
from rich.console import Console
from truss_chains import utils


class RAG(chains.StubBase):
    """Treat the whisper model like an external third party tool."""

    def run(
        self,
        query: str,
    ) -> str:
        json_payload = {"query": query, "params": {"num_context_docs": 3}}
        resp = self._remote.predict_sync(json_payload)
        return resp


@click.command()
@click.argument("query", type=str)
@click.option(
    "--url",
    default="https://model-4w7714dw.api.baseten.co/development/predict",
)
def main(query, url):
    rag = RAG.from_url_and_key(url, api_key=utils.get_api_key_from_trussrc())
    text = rag.run(query)
    console = Console()
    console.print(text)


if __name__ == "__main__":
    main()
