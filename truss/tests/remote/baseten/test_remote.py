import click
import pytest
from truss.remote.baseten.remote import BasetenRemote


def test_get_matching_version():
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]
    assert BasetenRemote._get_matching_version(versions, published=True)["id"] == "2"
    assert BasetenRemote._get_matching_version(versions, published=False)["id"] == "3"


def test_get_matching_version_dev_error():
    versions = [
        {"id": "1", "is_draft": False, "is_primary": True},
    ]
    with pytest.raises(click.UsageError):
        BasetenRemote._get_matching_version(versions, published=False)


def test_get_matching_version_prod_error():
    versions = [
        {"id": "1", "is_draft": True, "is_primary": False},
    ]
    with pytest.raises(click.UsageError):
        BasetenRemote._get_matching_version(versions, published=True)
