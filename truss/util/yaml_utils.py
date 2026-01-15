from typing import IO, Any

import yaml

# PyYAML's safe_load silently allows duplicate keys, using the last value.
# This can cause confusing bugs when a user accidentally defines the same key twice.
# This loader raises an error on duplicate keys to catch these issues early.


class _SafeLoaderNoDuplicateKeys(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key '{key}'",
                    key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep)


def safe_load_yaml_with_no_duplicates(stream: IO[str]) -> Any:
    return yaml.load(stream, Loader=_SafeLoaderNoDuplicateKeys)
