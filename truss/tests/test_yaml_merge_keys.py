import io
import warnings

import pytest
import yaml

from truss.util.yaml_utils import safe_load_yaml_with_no_duplicates


@pytest.mark.parametrize(
    "source",
    [
        "base: &base {cpu: 1, memory: 2}\nresources: {<<: *base}\n",
        "base: &base {cpu: 1}\nresources: {<<: *base, cpu: 2}\n",
        "base: &base {cpu: 1}\nresources: {cpu: 2, <<: *base}\n",
        "a: &a {cpu: 1}\nb: &b {cpu: 2, memory: 4}\nresources: {<<: [*a, *b]}\n",
        "a: &a {cpu: 1}\nb: &b {<<: *a, memory: 4}\nresources: {<<: *b}\n",
        "resources: {<<: {cpu: 1, memory: 2}}\n",
        "base: &base {cpu: 1}\nleft: {<<: *base}\nright: {<<: *base, cpu: 3}\n",
        "base: &base {cpu: 1}\nitems: [{<<: *base}, {<<: *base, cpu: 2}]\n",
        "base: &base {}\nresources: {<<: *base}\n",
    ],
)
def test_merge_keys_keep_safe_loader_semantics_without_duplicate_warnings(source):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = safe_load_yaml_with_no_duplicates(io.StringIO(source))
    assert actual == yaml.safe_load(source)
    assert recorded == []


def test_explicit_duplicates_still_warn_after_merge():
    source = "base: &base {cpu: 1}\nresources: {<<: *base, cpu: 2, cpu: 3}\n"
    with pytest.warns(UserWarning, match="duplicate key `cpu`"):
        actual = safe_load_yaml_with_no_duplicates(io.StringIO(source))
    assert actual["resources"]["cpu"] == 3


def test_explicit_duplicates_without_merges_still_warn():
    with pytest.warns(UserWarning, match="duplicate key `cpu`"):
        actual = safe_load_yaml_with_no_duplicates(io.StringIO("cpu: 1\ncpu: 2\n"))
    assert actual == {"cpu": 2}


def test_quoted_merge_spelling_remains_a_literal_key():
    source = "'<<': first\n'<<': second\n"
    with pytest.warns(UserWarning, match="duplicate key `<<`"):
        actual = safe_load_yaml_with_no_duplicates(io.StringIO(source))
    assert actual == {"<<": "second"}


@pytest.mark.parametrize("source", ["x: {<<: 1}", "x: {<<: [1]}", "x: {<<: null}"])
def test_invalid_merge_values_are_rejected_by_safe_loader(source):
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_load_yaml_with_no_duplicates(io.StringIO(source))


def test_unsafe_python_tag_is_not_enabled():
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_load_yaml_with_no_duplicates(
            io.StringIO("!!python/object:builtins.object {}")
        )


def test_caller_owned_stream_remains_open():
    stream = io.StringIO("base: &base {cpu: 1}\nresources: {<<: *base}\n")
    safe_load_yaml_with_no_duplicates(stream)
    assert not stream.closed
