## Contributing

Truss was first created at [Baseten](https://baseten.co), but as an open and living project eagerly accepts contributions of all kinds from the broader developer community. Please note that all participation with Truss falls under our [code of conduct](CODE_OF_CONDUCT.md).

We use GitHub features for project management on Truss:

* For bugs and feature requests, file an issue.
* For changes and updates, create a pull request.
* To view and comment on the roadmap, [check the projects tab](https://github.com/orgs/basetenlabs/projects/3).

## Local development

To get started contributing to the library, all you have to do is clone this repository!

### Setup

**PLEASE NOTE:** the ML ecosystem in general is still not well supported on M1 Macs, and as such, we do not recommend or support local development on M1 for Truss. Truss is well-optimized for use with GitHub Codespaces and other container-based development environments.

We use `asdf` to manage Python binaries and `poetry` to manage Python dependencies.

For development in a macOS environment, we use `brew` to manage system packages.

```
# Install asdf (or use another method https://asdf-vm.com/)
brew install asdf

# Install `asdf` managed python and poetry
asdf plugin add python
asdf plugin add poetry

# Install poetry dependencies
poetry install

# And finally precommit
poetry run pre-commit install
```

Then to run the entire test suite

```
poetry run pytest truss/tests
```

### Release

When releasing a version of the library with user-facing changes, be sure to update the [changelog](docs/CHANGELOG.md) with an overview of the changes, along with updating any relevant documentation. Feel free to tag @philipkiely-baseten to write or review any changelog or docs updates.
To release a new version of the library.

1. Create a PR changing the `pyproject.toml` version
2. Merge the PR, github actions will auto deploy if it detects the change

Steps to release a new verison of Truss to PyPi:

1. Ensure that the version in `pyproject.toml` matches the version number that we want to release
2. Create a PR from `main` into `release`, https://github.com/basetenlabs/truss/compare/release...main
3. After getting a review, merge the PR. After this, a new release will be created and automatically pushed to PyPi.


## Documentation

To learn about Truss see the [official documentation](https://truss.baseten.co).

Contributions to documentation are very welcome! Simply edit the appropriate markdown files in the `docs/` folder and make a pull request. For larger changes, tutorials, or any questions please contact [team@trussml.com](mailto:team@trussml.com).
