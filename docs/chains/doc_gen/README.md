This generation process of the documentation is *extremely* scrappy and just
an interim solution. It requires significant manual oversight and the code
quality in this directory is non-existent.

Extra deps required:
`pip install sphinx sphinx_rtd_theme sphinx_markdown_builder sphinx-pydantic`


The general process is:
1. Document as much as possible in the code, including usage examples, links
   etc.
2. Auto-generate `generated-API-reference.mdx` with
   `uv run python docs/chains/doc_gen/generate_reference.py`.
   This applies the patch file and launches meld to resolve conflicts.
4. Proofread `docs/chains/doc_gen/API-reference.mdx`.
5. If proofreading leads to edits or the upstream docstrings changed lot,
   update the patch file: `diff -u \
   docs/chains/doc_gen/generated-reference.mdx \
   docs/chains/doc_gen/API-reference.mdx > \
   docs/chains/doc_gen/reference.patch`

For questions, please reach out to @marius-baseten.
