# Request & Response Serialization

The default path `v1/models/model:predict` will consume and produce JSON with
some liberties taken for to support more complex types.

The path `v1/models/model:predict_binary` will consume and produce a BaseTen
opinionated binary serialization. It is largely based on `msgpack` and can be
seen at `templates/common/serialization.py`. This will work natively
inside a BaseTen worklet.
