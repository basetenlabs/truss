# Control Truss

Control Trusses are Trusses with a control plane. Currently, they are meant to
sync changes to a Truss into a running container. Avoiding the building of
docker image and spinning up of docker container can result in a much faster
development loop.

Control web-server uses its own virtual environment so as not to interfere with
python requirements of the inference server.

## Config

A Truss can be indicated to be a control Truss by setting use_control_plane
property to True in `config.yaml`.

## Architecture

Control Trusses run an additional control web-server process in addition to the
inference server process. The control web-server acts as a proxy to the
inference web-server. Additionlly, it provides a patch endpoint that can be used
to apply and load changes into the inference web-server.

Imagine that a Control Truss is running on docker. Say changes are made to this
Truss and `docker_predict` is called on the handle. `docker_predict` just
patches the running Truss container by calling the patch endpoing on the control
web-server and avoids building and deploying a new image.


## Patch computation

To understand patch computation it's important to understand a Truss's hash and
signature.

### Truss Hash

A Truss's hash is the sha256 hash of the Truss's content.

### Truss Signature
A Truss's signature is a form of digest that is meant for computing changes to a
truss from a previous state. The idea is that high level changes can be gleaned
from a short previous representation rather than the whole Truss. This avoids
storing around previous versions of the Truss, drastically reducing space requirements.

Currently, the signature consists of two things:
1. All the filepaths under the Truss and their content hashes
2. Contents of `config.yaml`

Signature is expected to be just a few Kbs for most cases.

### Patch

A patch represents a change to the Truss. Examples are changes to:
1. Model code
2. Python requirements
3. System packages
4. Environment variables

Currently, only model code changes are supported, but other changes are planned
to be supported soon.
