FROM baseten/truss-context-builder:test

COPY test.py test.py

# This seems to be present on github actions execution environment and interferes.
# So, get rid of it.
RUN rm -rf /.venv

RUN poetry run python test.py
