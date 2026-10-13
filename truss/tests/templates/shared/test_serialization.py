import datetime
import uuid
from decimal import Decimal

import numpy as np
import pytest

from truss.templates.shared import serialization


def test_roundtrip_numeric_numpy_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    restored = serialization.truss_msgpack_deserialize(
        serialization.truss_msgpack_serialize(arr)
    )
    np.testing.assert_array_equal(restored, arr)


def test_roundtrip_extended_scalar_types():
    obj = {
        "dt": datetime.datetime(2026, 1, 2, 3, 4, 5),
        "date": datetime.date(2026, 1, 2),
        "decimal": Decimal("1.5"),
        "uuid": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    }
    restored = serialization.truss_msgpack_deserialize(
        serialization.truss_msgpack_serialize(obj)
    )
    assert restored == obj


def test_object_array_payload_is_rejected():
    # An object-dtype array is what triggers the (unsafe) pickle path in
    # msgpack_numpy's decoder; serializing one here produces exactly the
    # nd/kind markers a malicious client would send.
    obj_array = np.array([{"a": 1}, "b"], dtype=object)
    payload = serialization.truss_msgpack_serialize(obj_array)

    with pytest.raises(ValueError, match="unsupported payload"):
        serialization.truss_msgpack_deserialize(payload)


def test_nested_object_array_payload_is_rejected():
    # The guard runs via object_hook, so an object array nested inside a
    # larger structure must be rejected too, not just a top-level one.
    payload = serialization.truss_msgpack_serialize(
        {"inputs": np.array(["x"], dtype=object)}
    )

    with pytest.raises(ValueError, match="unsupported payload"):
        serialization.truss_msgpack_deserialize(payload)
