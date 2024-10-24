# Setting up local development (contributor)

To get started contributing to Truss, first fork the repository.

## Truss setup

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
poetry install --with=dev --extras=all

# And finally precommit
poetry run pre-commit install
```

Then to run the entire test suite

```
poetry run pytest truss/tests
```

## Docs setup

Contributions to documentation are very welcome! Simply edit the appropriate markdown files in the `docs/` folder and make a pull request. For larger changes, tutorials, or any questions please contact [team@trussml.com](mailto:team@trussml.com).

Baseten docs are built using Mintlify. To run the docs site locally, use Mintlify's [getting started guide](https://mintlify.com/docs/development).
