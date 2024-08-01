from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrussSignature:
    """Truss signature stores information for calculating patches for future
    changes to Truss.

    Currently, it stores hashes of all of the paths in the truss directory excluding the data dir,
    and the truss config contents. Path hashes allow calculating added/updated/removes
    paths in future trusses compared to this. Config contents allow calculating
    config changes, such as add/update/remove of python requirements etc.
    """

    content_hashes_by_path: Dict[str, str]
    config: str
    requirements_file_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content_hashes_by_path": self.content_hashes_by_path,
            "config": self.config,
            "requirements_file_requirements": self.requirements_file_requirements,
        }

    @staticmethod
    def from_dict(d) -> "TrussSignature":
        return TrussSignature(
            content_hashes_by_path=d["content_hashes_by_path"],
            config=d["config"],
            requirements_file_requirements=d.get("requirements_file_requirements", []),
        )


ChangedPaths = Dict[str, List[str]]
