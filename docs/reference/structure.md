# Truss directory structure

The files in a Truss and their contents

## config.yaml


## examples.yaml


## model/


### __init__.py

Always empty, generated only for Python packaging purposes.

### model.py

Implements the class `Model`

## data/

Most hetergeneous contents, optional, won't be totally enumerated (beyond scope)

Contains the dependencies for the model like weights, parameters, anything else from exporting an in-memory model

Could be pretty big if you have a 100 billion parameter model