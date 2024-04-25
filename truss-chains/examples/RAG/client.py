import logging

logging.basicConfig(level=logging.WARNING)

import re

import click
import requests
import truss_chains as chains
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
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

    # patterns = {
    #     "python": r"```python\n(.*?)\n```",
    #     "json": r"```json\n(.*?)\n```",
    #     "bash": r"```bash\n(.*?)\n```",
    #     "bash": r"```sh\n(.*?)\n```",
    #     "bash": r"```shell\n(.*?)\n```",
    # }
    #
    #
    #
    # def find_language(pattern_matched: str) -> str:
    #     for language, pattern in patterns.items():
    #         if re.fullmatch(pattern, pattern_matched):
    #             return language
    #     return "text"  # Default to text if no match is found
    #
    # last_end = 0
    # for match in re.finditer("|".join([v for v in patterns.values()]), text, re.DOTALL):
    #     start, end = match.span()
    #     console.print(text[last_end:start], end="")
    #
    #     language = find_language(match.group())
    #     code = match.group(1).strip()
    #     syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    #     console.print(syntax)
    #
    #     last_end = end
    #
    # console.print(text[last_end:], end="")


if __name__ == "__main__":
    main()
