# Bundled packages

If your model depends on internal packages that are not available on PyPi, you can bundle this code with the Truss in the `packages` folder. These packages then become available as top level modules to the model class's code when it executes, whether it is running locally, in any Docker container, or on Baseten.

{% hint style="info" %}
When you import your Truss, the import mechanism add the packages in the Truss' `packages` directory to the path.
{% endhint %}

Great care should be taken to avoid conflicts between these packages with any
python requirements. Note that the serving environment may itself be dependent
on some standard Python modules such as `requests`, so it's best to
avoid using package names that may conflict with any standard or popular Python
packages.

## External packages

There are situations where you may need to maintain code separately from your Truss. For example, you may want to share code across multiple Trusses without publishing it to PyPi. External packages are built for this use case.

{% hint style="info" %}
Take a look at [this demo repository](https://github.com/bolasim/truss-packages-example) on Truss external packages.
{% endhint %}
### Config

A Truss can be provided a set of external folders as places to look up local
packages, using the `external_package_dirs` setting in `config.yaml`. This settings is similar to
`bundled_packages_dir` except that these folders can be outside of the Truss
folder. These external directories are then effectively available to the model
and training python code to lookup modules and packages from.

### Terminology

A Truss that has any external packages is considered `scattered`, because the
Truss is not self contained. Correspondingly, `gather` means collecting into a
self contained Truss. `TrussHandle` has a method called `gather` to gather a
scattered  Truss.

Note that a scattered Truss will likely not work on another machine, unless all
the `external_package_dirs` are carefully replicated there as well. For example, this will
work for Git repos that contain both the Truss and the `external_package_dirs`
and relative paths are used. Where possible, a scattered Truss should be gathered
into an equivalent Truss using the `gather` operation, before sending over.

### Implementation

For in-memory serving and training, i.e. without creating and running a Docker
image, these external directories are added to sys.path and thus become
available to model/training code.

For any Docker based execution, a gathered Truss is first created and then
executed. This gathered Truss works behind the scenes and the user doesn't need
to touch it directly. The gathered Truss is cached and reused, and only
recreated if any part of the original Truss is modified. This flow will likely
be optimized in future to only update the changed parts, rather than recreate.
