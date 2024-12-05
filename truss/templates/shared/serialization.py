import json
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from numpy.typing import NDArray


JSONType = Union[str, int, float, bool, None, List["JSONType"], Dict[str, "JSONType"]]
MsgPackType = Union[
    str,
    int,
    float,
    bool,
    None,
    date,
    Decimal,
    datetime,
    time,
    timedelta,
    uuid.UUID,
    "NDArray",
    List["MsgPackType"],
    Dict[str, "MsgPackType"],
]


# mostly cribbed from django.core.serializer.DjangoJSONEncoder
def _truss_msgpack_encoder(
    obj: Union[Decimal, date, time, timedelta, uuid.UUID, Dict],
    chain: Optional[Callable] = None,
) -> Dict:
    if isinstance(obj, datetime):
        r = obj.isoformat()
        if r.endswith("+00:00"):
            r = r[:-6] + "Z"
        return {b"__dt_datetime_iso__": True, b"data": r}
    elif isinstance(obj, date):
        r = obj.isoformat()
        return {b"__dt_date_iso__": True, b"data": r}
    elif isinstance(obj, time):
        if obj.utcoffset() is not None:
            raise ValueError("Cannot represent timezone-aware times.")
        r = obj.isoformat()
        return {b"__dt_time_iso__": True, b"data": r}
    elif isinstance(obj, timedelta):
        return {
            b"__dt_timedelta__": True,
            b"data": (obj.days, obj.seconds, obj.microseconds),
        }
    elif isinstance(obj, Decimal):
        return {b"__decimal__": True, b"data": str(obj)}
    elif isinstance(obj, uuid.UUID):
        return {b"__uuid__": True, b"data": str(obj)}
    else:
        return obj if chain is None else chain(obj)


def _truss_msgpack_decoder(obj: Any, chain=None):
    try:
        if b"__dt_datetime_iso__" in obj:
            return datetime.fromisoformat(obj[b"data"])
        elif b"__dt_date_iso__" in obj:
            return date.fromisoformat(obj[b"data"])
        elif b"__dt_time_iso__" in obj:
            return time.fromisoformat(obj[b"data"])
        elif b"__dt_timedelta__" in obj:
            days, seconds, microseconds = obj[b"data"]
            return timedelta(days=days, seconds=seconds, microseconds=microseconds)
        elif b"__decimal__" in obj:
            return Decimal(obj[b"data"])
        elif b"__uuid__" in obj:
            return uuid.UUID(obj[b"data"])
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


# this json object is JSONType + np.array + datetime
def is_truss_serializable(obj: Any) -> bool:
    import numpy as np

    # basic JSON types
    if isinstance(obj, (str, int, float, bool, type(None), dict, list)):
        return True
    elif isinstance(obj, (datetime, date, time, timedelta)):
        return True
    elif isinstance(obj, np.ndarray):
        return True
    else:
        return False


def truss_msgpack_serialize(obj: MsgPackType) -> bytes:
    import msgpack
    import msgpack_numpy as mp_np

    return msgpack.packb(
        obj, default=lambda x: _truss_msgpack_encoder(x, chain=mp_np.encode)
    )


def truss_msgpack_deserialize(data: bytes) -> MsgPackType:
    import msgpack
    import msgpack_numpy as mp_np

    return msgpack.unpackb(
        data, object_hook=lambda x: _truss_msgpack_decoder(x, chain=mp_np.decode)
    )


class DeepNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
