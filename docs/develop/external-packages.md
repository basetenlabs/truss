# External packages

Where possible, a truss folder should be self contained. But, there are certain
situations where you may need to keep some code outside. e.g. when you want
to share code across multiple local trusses and publishing that code into a pypi
package is not an option. External packages are meant for this use case, but
should be used carefully.

{% hint style="info" %}
Take a look at this demo repository on Truss external packages: [https://github.com/bolasm/truss-packages-example](https://github.com/bolasm/truss-packages-example)
{% endhint %}
# Config

A truss can be provided a set of external folders as places to look up local
packages, using the `external_package_dirs` setting. This settings is similar to
`bundled_packages_dir` except that these folders can be outside of the truss
folder. These external directories are then effectively available to the model
and training python code to lookup modules and packages from.

# Terminology

A truss that has any external pacakges is considered `scattered`, because the
truss is not self contained. Correspondingly, `gather` means collecting into a
self contained truss. `TrussHandle` has a method called `gather` to gather a
scattered  truss.

Note that a scattered Truss will likley not work on another machine, unless all
the `external_package_dirs` are carefully replicated there as well. e.g. this will
work for git repos that contain both the truss and the `external_package_dirs`
and relative paths are used. Where possible, a scattered truss should be gathered
into an equivalent truss using the `gather` operation, before sending over.

# Implementation

For in-memory serving and training, i.e. without creating and running a docker
image, these external directories are added to sys.path and thus become
available to model/training code.

For any docker based execution, a gathered truss is first created and then
executed. This gathered truss works behind the scenes and the user doesn't need
to touch it directly. The gathered truss is cached and reused, and only
recreated if any part of the original truss is modified. This flow will likely
be optimized in future to only update the changed parts, rather than recreate.
