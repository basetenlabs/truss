## Contributing

Truss was first created at [Baseten](https://baseten.co), but as an open and living project eagerly accepts contributions of all kinds from the broader developer community. Please note that all participation with Truss falls under our [code of conduct](CODE_OF_CONDUCT.md).

We use GitHub features for project management on Truss:

* For bugs and feature requests, file an issue.
* For changes and updates, create a pull request.
* To view and comment on the roadmap, [check the projects tab](https://github.com/orgs/basetenlabs/projects/3).

## Local development

To get started contributing to the library, all you have to do is clone this repository!

### Setup
We use [`rye`](https://rye-up.com/) to manage Python environments, packaging (via [`hatchiling`](https://hatch.pypa.io/latest/)), and depdencies.

For development:

```bash
# Install rye: https://rye-up.com/guide/installation/#installing-rye
curl -sSf https://rye-up.com/get | bash

# Add to shell
echo 'source "$HOME/.rye/env"' >> ~/.bashrc

# Setup environment (includes python version and dependencies)
rye sync
```

Then to run the entire test suite

```bash
rye run pytest truss/tests
```
### Build Wheel for release
TODO: thils will automated in CI soon
```bash
rye build --wheel --pyproject ./pyproject.toml --clean --verbose
```

## Documentation

To learn about Truss see the [official documentation](https://truss.baseten.co).

Contributions to documentation are very welcome! Simply edit the appropriate markdown files in the `docs/` folder and make a pull request. For larger changes, tutorials, or any questions please contact [hi@baseten.co](mailto:hi@baseten.co).
