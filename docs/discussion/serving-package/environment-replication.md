# Environment Replication

The default `mk_truss` factory methods do inference on the Python runtime
environment for required libraries. This means that it introspects the Python
runtime and infers the versions of required libraries for respective framework
and local environment then populates the `requirements.txt` with the inferred
definitions. Any subsequent extension beyond what is inferred should be added to
`requirements.txt`
