import datetime
import decimal
import json
import uuid

import msgpack
import msgpack_numpy as mp_np
import numpy as np


# mostly cribbed from django.core.serializer.DjangoJSONEncoder
def truss_msgpack_encoder(obj, chain=None):
    if isinstance(obj, datetime.datetime):
        r = obj.isoformat()
        if r.endswith('+00:00'):
            r = r[:-6] + 'Z'
        return {b'__dt_datetime_iso__': True, b'data': r}
    elif isinstance(obj, datetime.date):
        r = obj.isoformat()
        return {b'__dt_date_iso__': True, b'data': r}
    elif isinstance(obj, datetime.time):
        if obj.utcoffset() is not None:
            raise ValueError("Cannot represent timezone-aware times.")
        r = obj.isoformat()
        return {b'__dt_time_iso__': True, b'data': r}
    elif isinstance(obj, datetime.timedelta):
        return {b'__dt_timedelta__': True, b'data': (obj.days, obj.seconds, obj.microseconds)}
    elif isinstance(obj, decimal.Decimal):
        return {b'__decimal__': True, b'data': str(obj)}
    elif isinstance(obj, uuid.UUID):
        return {b'__uuid__': True, b'data': str(obj)}
    else:
        return obj if chain is None else chain(obj)


def truss_msgpack_decoder(obj, chain=None):
    try:
        if b'__dt_datetime_iso__' in obj:
            return datetime.datetime.fromisoformat(obj[b'data'])
        elif b'__dt_date_iso__' in obj:
            return datetime.date.fromisoformat(obj[b'data'])
        elif b'__dt_time_iso__' in obj:
            return datetime.time.fromisoformat(obj[b'data'])
        elif b'__dt_timedelta__' in obj:
            days, seconds, microseconds = obj[b'data']
            return datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds)
        elif b'__decimal__' in obj:
            return decimal.Decimal(obj[b'data'])
        elif b'__uuid__' in obj:
            return uuid.UUID(obj[b'data'])
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


# this json object is JSONType + np.array + datetime
def is_truss_serializable(obj):
    # basic JSON types
    if isinstance(obj, (str, int, float, bool, type(None), dict, list)):
        return True
    elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time, datetime.timedelta)):
        return True
    elif isinstance(obj, np.ndarray):
        return True
    else:
        return False


def truss_msgpack_serialize(obj):
    return msgpack.packb(obj, default=lambda x: truss_msgpack_encoder(x, chain=mp_np.encode))


def truss_msgpack_deserialize(obj):
    return msgpack.unpackb(obj, object_hook=lambda x: truss_msgpack_decoder(x, chain=mp_np.decode))


class DeepNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(DeepNumpyEncoder, self).default(obj)
