# ROADMAP

Truss is a living, breathing project. Our goal is to make it as easy as possible
to get value from your work by providing a simple, yet powerful way to package
and deploy machine learning models into production systems. We're also building
a community of developers who are passionate about building machine learning
systems. Our roadmap is a way to keep you informed about the progress of our
project, and what we're working on.

We have a few milestones we're working on, and we're always open to feedback on
our roadmap. Our goal is to support data scientists and machine learning
engineers by meeting them at the point where they're done with their work and
ready to move on to the next step. The next step is almost always to provide a
way for their model to be consumed by other systems. But there are other areas
where we can support the deployment of models, and we're investing in those
areas too.

Our primary goal areas for the roadmap are:

- Model support
- Testing and reliability
- Performance
- Model support modules

## Model support

Today truss supports a wide variety of machine learning models. We're working on
a few new ones, and we're also working on extending support for the existing
ones. If you have a model that you'd like to see supported, please let us know
by [filing a feature request](https://github.com/basetenlabs/truss/issues).

### New frameworks

- ONNX
- R
- Apache MXNet

## Testing and reliability

One of the key issues plaguing machine learning systems is the need to test
them. We're working on a way to make it easier to test that your models and data
are working as expected. Currently we support smoke testing prediction flows via
in-memory and dockerized environments. By writing examples we can verify the
model is working as expected.

Our goal is to support testing assertions on both the model and the data.

## Performance

### Deployment backends

Model deployments are a critical part of machine learning systems. We're working
on a way to make it easier to deploy models on a variety of platforms. Each
platform has its own set of performance tradeoffs. Today our model backend is a
tornado server based heavily on kserve's implementation, and we're working on a
way to make it easier to deploy models on other backends. This will allow truss
users to exchange tornado for other API backends.

### GPU support

Many models benefit from GPU training and inference. There are a few ways in
which we can optimize support for GPUs at inference time:

- Multi-GPU support
- GPU sharing

Multi-GPU support will allow truss users to deploy larger models and inference
parallelism. GPU sharing will allow users to share GPUs between models.

## Model support modules

Alongside any given model deployment, it's often desirable to perform some side
actions. For example model monitoring, health checks, logging, anomaly
detection, and explainability are all examples of side actions. We're
formulating a way to make it easier to perform these side actions by creating
modular components that can be used as needed by the model deployment.

---

If you'd like to contribute to the project, please join our community. If you'd
like to see a feature added to the project, please [file a feature
request](https://github.com/basetenlabs/truss/issues) in Issues.
