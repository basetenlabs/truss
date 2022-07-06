---
description: Containers for serving Machine Learning Models
---

# What is a Truss?

A Truss is a context for building a container for serving predictions from a
model. Trusses are designed to work seamlessly with in-memory models from
supported model frameworks while maintaining the ability to serve predictions
for more complex scenarios. Trusses can be created local to the environment of
the client for introspection and any required debugging and then when ready,
uploaded in our serving environment or onto another container serving platform
