# Local model development



The point of local development is to test pre- and post-processing. Deployment loop is very slow. Instead, use fast local iteration to make sure it works, and docker means that the environments matches production very closely.


## Staging checklist

Use Truss locally to validate that your model works on an environment that, thanks to Docker, closely resembles production.

Here are some things you might want to check:

* Dependencies: Are the Python packages you need available and working?
* System packages: Are system-level dependencies installed with the correct version?
* Environment variables: Do you have all of your config vars and API keys securely