FROM baseten/baseten-server-base-py39:latest

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install --yes --no-install-recommends git fzf vim curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -L -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz \
    && tar -xvf git-lfs.tar.gz \
    && mv git-lfs /usr/local/bin/git-lfs \
    && rm git-lfs.tar.gz \
    && chmod 0755 /usr/local/bin/git-lfs

RUN curl -sSL https://install.python-poetry.org | python
ENV PATH=/home/vscode/.poetry/bin:$PATH

CMD [ "sleep", "infinity" ]
