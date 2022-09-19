FROM baseten/baseten-server-base-py39:latest

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install --yes --no-install-recommends git fzf vim \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -sSL https://install.python-poetry.org | python
ENV PATH=/home/vscode/.poetry/bin:$PATH

CMD [ "sleep", "infinity" ]
