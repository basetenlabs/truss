# Creating a release

<Note>
You must be a repository admin to create a PyPi release or release candidate.
</Note>

To release a new version of the library.

1. Create a PR changing the `pyproject.toml` version
2. Merge the PR, github actions will auto deploy if it detects the change

Steps to release a new verison of Truss to PyPi:

1. Ensure that the version in `pyproject.toml` matches the version number that we want to release
2. Create a PR from `main` into `release`, https://github.com/basetenlabs/truss/compare/release...main
3. After getting a review, merge the PR. After this, a new release will be created and automatically pushed to PyPi.
