from dataclasses import dataclass
from typing import Dict


@dataclass
class TrussSignature:
    content_hashes_by_path: Dict[str, str]
    config: str

    def to_dict(self) -> dict:
        return {
            "content_hashes_by_path": self.content_hashes_by_path,
            "config": self.config,
        }

    @staticmethod
    def from_dict(d) -> "TrussSignature":
        return TrussSignature(
            content_hashes_by_path=d["content_hashes_by_path"], config=d["config"]
        )
