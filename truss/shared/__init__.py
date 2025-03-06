import pydantic

pydantic_major_version = int(pydantic.VERSION.split(".")[0])
if pydantic_major_version < 2:
    raise RuntimeError(
        "Please upgrade to pydantic v2 to use shared chains / train features"
    )

del pydantic, pydantic_major_version
