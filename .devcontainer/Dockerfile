ARG VARIANT="3.9-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/

ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="false"
ARG USE_MOBY="true"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && /bin/bash /tmp/library-scripts/docker-in-docker-debian.sh "true" "${USERNAME}" "${USE_MOBY}" \
    && /bin/bash /tmp/library-scripts/baseten.sh "true" "${USERNAME}" "${USE_MOBY}" \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install --yes --no-install-recommends git fzf vim \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

USER vscode
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH=/home/vscode/.poetry/bin:$PATH

USER root

VOLUME [ "/var/lib/docker" ]

# Setting the ENTRYPOINT to docker-init.sh will start up the Docker Engine
# inside the container "overrideCommand": false is set in devcontainer.json.
# The script will also execute CMD if you need to alter startup behaviors.
ENTRYPOINT [ "/usr/local/share/docker-init.sh" ]
CMD [ "sleep", "infinity" ]
