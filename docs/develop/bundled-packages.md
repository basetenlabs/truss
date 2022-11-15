# Bundled packages

If your model depends on internal packages that are not available on PyPi, you can bundle this code with the Truss in the `packages` folder. These packages then become available as top level modules to the model class's code when it executes, whether it is running locally, in any Docker container, or on Baseten.

{% hint style="info" %}
When you import your Truss, the import mechanism add the packages in the Truss' `packages` directory to the path.
{% endhint %}

Great care should be taken to avoid conflicts between these packages with any
python requirements. Note that the serving environment may itself be dependent
on some standard Python modules such as `requests` and `kserve`, so it's best to
avoid using package names that may conflict with any standard or popular Python
packages.
