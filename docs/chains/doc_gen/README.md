# Generate the Chains reference

The generator reads the installed `truss_chains` API, renders MDX with Sphinx,
and applies `reference.patch` to preserve curated descriptions and links.

Run the commands from this checkout's root so `uv` uses its package source.
The curation patch tracks this checkout's docstrings; an installed release
with different docstrings can fail to apply the patch.

Install the documentation dependencies in your project environment:

```sh
uv pip install "sphinx==7.4.7" sphinx_rtd_theme "sphinx-markdown-builder==0.6.6" sphinx-pydantic
```

Use these Sphinx and Markdown builder versions to preserve MDX output and
parameter types. Verify the package or source version before regenerating.

The MDX adapter normalizes the extra type argument that Python 3.13 renders
for annotations like `typing.ContextManager[None]` (`[None, bool | None]` on
3.13, `[None]` earlier), so regenerated output is byte-identical on Python
3.9 through 3.13. If a future Python version changes generic-rendering
defaults again, `--check` reports the drift; extend the normalization in
`mdx_adapter.py` rather than regenerating with different output.

To regenerate both reference files, run:

```sh
uv run python docs/chains/doc_gen/generate_reference.py
```

The generator writes `generated-reference.mdx` and `API-reference.mdx` only
after the curation patch applies. A stale patch fails without opening an
interactive editor or replacing the existing reference files.

To check the committed output without replacing either file, run:

```sh
uv run python docs/chains/doc_gen/generate_reference.py --check
```

Run the adapter and failure-handling regression tests in the documentation
environment. These require the Sphinx dependencies above, which the application
test environment does not install:

```sh
uv run python -m unittest discover -s docs/chains/doc_gen -p 'regression_*.py'
```

When source docstrings or signatures change, export the uncurated output:

```sh
uv run python docs/chains/doc_gen/generate_reference.py --raw-out /tmp/chains-reference.mdx
```

Review that output against `API-reference.mdx` and preserve its curated prose.
After updating the curated file, replace `generated-reference.mdx` with the
reviewed raw output.
Refresh the patch after reviewing the complete output:

```sh
diff -u docs/chains/doc_gen/generated-reference.mdx \
  docs/chains/doc_gen/API-reference.mdx > docs/chains/doc_gen/reference.patch
```

`diff` exits with status 1 when the files differ. Regenerate and run `--check`
after refreshing the patch. Questions about curation: @marius-baseten.
