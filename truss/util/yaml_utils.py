import warnings
from typing import IO, Any

import yaml


class _SafeLoaderWarnDuplicateKeys(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            # TODO around ~7/2026: Change this to an error once we've given users time to fix their configs.
            if key in keys:
                warnings.warn(
                    f"Detected duplicate key `{key}`, will use the last entry but could cause unexpected behavior"
                )
            keys.add(key)
        return super().construct_mapping(node, deep)


def safe_load_yaml_with_no_duplicates(stream: IO[str]) -> Any:
    return yaml.load(stream, Loader=_SafeLoaderWarnDuplicateKeys)
