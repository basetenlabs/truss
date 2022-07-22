# Bundled packages

There are times when model code depends on internal packages that are not
available on pypi. This code can be bundled as submodule under the model module
but that may not work all the time. e.g. the in-memory model may refer to these
packages as top level modules and that may not work when they become submodules.
Bundled packages can help in that scenario.

Top level packages can be bundled with the truss by placing them in the
`packages` folder under the truss directory. These packages then become
available as top level modules to the model class's code when it executes in the
local, docker or baseten deployed environments.

Great care should be taken to avoid conflicts between these packages with any
python requirements. Note that the serving environment may itself be dependent
on some standard python modules such as `requests` and `kserve`, so it's best to
avoid using package names that may conflict with any standard or popular python
packages.
