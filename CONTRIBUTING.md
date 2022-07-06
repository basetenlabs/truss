## Contributing

We are looking to grow this project, get more contributors, and extend functionality for more model frameworks. We are
also looking for user feedback on the mechanisms of the library.

Kindly submit any bugs and feature requests as issues on Github. Please feel free to submit any pull requests with a
reference to a Github issue.

## Getting help

Please contact us on our channel here (some link)

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

For more information see the [official documentation](https://baseten.gitbook.io/baseten-scaffolds/)

## Contributors
* created by [Baseten Labs, Inc](http://baseten.co)
* main developers: Phil Howes, Alex Gillmor
* and user's like you!
