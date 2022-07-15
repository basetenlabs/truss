## Contributing

Truss was first created at [Baseten](https://baseten.co), but as an open and living project eagerly accepts contributions of all kinds from the broader developer community. Please note that all participation with Truss falls under our [code of conduct](CODE_OF_CONDUCT.md).

We use GitHub features for project management on Truss:

* For bugs and feature requests, file an issue.
* For changes and updates, create a pull request.
* To view and comment on the roadmap, check the projects tab.

## Local development

To get started contributing to the library, all you have to do is clone this repository!

### Setup

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

## Documentation

To learn about Truss see the [official documentation](https://truss.baseten.co).

Contributions to documentation are very welcome! Simply edit the appropriate markdown files in the `docs/` folder and make a pull request. For larger changes, tutorials, or any questions please contact [philip.kiely@baseten.co](mailto:philip.kiely@baseten.co)

## Contributors

Truss was made possible by:

[Baseten Labs, Inc](http://baseten.co)
* Phil Howes
* Alex Gillmor
* Pankaj Gupta
* Philip Kiely
* Nish Singaraju
* Abu Qadar
* and users like you!
