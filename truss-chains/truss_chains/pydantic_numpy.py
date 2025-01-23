import base64
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NumpyArrayField:
    """Wrapper class to support numpy arrays as fields on pydantic models and provide
    JSON or binary serialization implementations.

    The JSON serialization exposes (data, shape, dtype), and the data is base-64
    encoded which leads to ~33% overhead. A more compact serialization can be achieved
    using ``msgpack_numpy`` (integrated in chains, if RPC-option ``use_binary`` is
    enabled).

    Usage example:

    ```
    import numpy as np

    class MyModel(pydantic.BaseModel):
        my_array: NumpyArrayField

    m = MyModel(my_array=np.arange(4).reshape((2, 2)))
    m.my_array.array += 10  # Work with the numpy array.
    print(m)
    # my_array=NumpyArrayField(
    #   shape=(2, 2),
    #   dtype=int64,
    #   data=[[10 11] [12 13]])
    m_json = m.model_dump_json()  # Serialize.
    print(m_json)
    # {"my_array":{"data_b64":"CgAAAAAAAAALAAAAAAAAAAwAAAAAAAAADQAAAAAAAAA=","shape":[2,2],"dtype":"int64"}}
    m2 = MyModel.model_validate_json(m_json)  # De-serialize.
    ```
    """

    data_key: ClassVar[str] = "data_b64"
    shape_key: ClassVar[str] = "shape"
    dtype_key: ClassVar[str] = "dtype"
    array: "NDArray"

    def __init__(self, array: "NDArray"):
        self.array = array

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={self.array.shape}, "
            f"dtype={self.array.dtype}, data={self.array})"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate_numpy_array,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize_numpy_array, info_arg=True
            ),
        )

    @classmethod
    def validate_numpy_array(cls, value: Any):
        import numpy as np

        keys = {cls.data_key, cls.shape_key, cls.dtype_key}
        if isinstance(value, dict) and keys.issubset(value):
            try:
                data = base64.b64decode(value[cls.data_key])
                array = np.frombuffer(data, dtype=value[cls.dtype_key]).reshape(
                    value[cls.shape_key]
                )
                return cls(array)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    "numpy_array_validation"
                    f"Invalid data, shape, or dtype for NumPy array: {str(e)}"
                )
        if isinstance(value, np.ndarray):
            return cls(value)
        if isinstance(value, cls):
            return value

        raise TypeError(
            "numpy_array_validation\n"
            f"Expected a NumPy array or a dictionary with keys {keys}.\n"
            f"Got:\n{value}"
        )

    @classmethod
    def serialize_numpy_array(
        cls, obj: "NumpyArrayField", info: core_schema.SerializationInfo
    ):
        if info.mode == "json":
            return {
                cls.data_key: base64.b64encode(obj.array.tobytes()).decode("utf-8"),
                cls.shape_key: obj.array.shape,
                cls.dtype_key: str(obj.array.dtype),
            }
        return obj.array

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(_core_schema)
        json_schema.update(
            {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "format": "byte"},
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "dtype": {"type": "string"},
                },
                "required": ["data", "shape", "dtype"],
            }
        )
        return json_schema
