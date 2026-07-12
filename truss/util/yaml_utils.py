import warnings
from typing import IO, Any

import yaml


class _SafeLoaderWarnDuplicateKeys(yaml.SafeLoader):
    strict: bool = False

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                if self.strict:
                    raise ValueError(f"Duplicate key `{key}` in YAML mapping")
                else:
                    warnings.warn(
                        f"Detected duplicate key `{key}`, will use the last entry but could cause unexpected behavior"
                    )
            keys.add(key)
        return super().construct_mapping(node, deep)


def safe_load_yaml_with_no_duplicates(stream: IO[str], strict: bool = False) -> Any:
    _SafeLoaderWarnDuplicateKeys.strict = strict
    return yaml.load(stream, Loader=_SafeLoaderWarnDuplicateKeys)
