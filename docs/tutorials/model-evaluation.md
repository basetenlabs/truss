The point of local development is to test pre- and post-processing. Deployment loop is very slow. Instead, use fast local iteration to make sure it works, and docker means that the environments matches production very closely.


Check dependencies, system packages, environment variables, etc locally. But it actually looks like prod, not your jupyter notebook.
