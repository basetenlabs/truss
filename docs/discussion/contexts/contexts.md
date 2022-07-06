# Contexts

Having the model packaged in a standard format allow for it to be used in
various contexts. A few of the important contexts are:

1. Load local This allows loading the model directly in the running process.
This assumes that the running process is already capable of executing the model,
e.g. has the needed python pacakges installed. This is a quick may of trying out
a model.
2. Image builder This allows building a docker image out of the model package.
This serving ready image provides a more reliable way of testing your model
locally. The image is ready to use for serving purposes.
3. Deploy to Baseten The model package can be passed to Baseten to serve the
model. Internally, Baseten also uses the image builder above but also adds some
monitoring on top.

A context is just a method that's passed a handle to the truss directory; it's
really easy to define one. We expect a lot more contexts to be defined in
future. e.g. A context may be used to convert the truss package to a different
format such as MLFlow or for exporting to serving environments such as
SageMaker. Any contributions regarding these or any others are most welcome.
