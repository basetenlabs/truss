import pydantic

pydantic_major_version = int(pydantic.VERSION.split(".")[0])
if pydantic_major_version < 2:
    raise RuntimeError(
        f"Pydantic version {pydantic.VERSION} is not supported for shared code."
    )

del pydantic, pydantic_major_version
