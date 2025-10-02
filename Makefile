docker_build_context_builder:
	VERSION=$(shell grep "^version " pyproject.toml | awk '{ print $$NF }' | sed 's/^"\(.*\)"$$/\1/') \
	&& docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64,linux/arm64 -t baseten/truss-context-builder:v$$VERSION

docker_build_push_context_builder: docker_build_context_builder
	VERSION=$(shell grep "^version " pyproject.toml | awk '{ print $$NF }' | sed 's/^"\(.*\)"$$/\1/') \
	&& docker push baseten/truss-context-builder:v$$VERSION

format:
	uv run ruff check --fix
	uv run ruff format
