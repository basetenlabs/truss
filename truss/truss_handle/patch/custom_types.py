from dataclasses import dataclass, field
from typing import Dict, List

from pydantic import BaseModel

from truss.templates.control.control.helpers.custom_types import Patch
from truss.util.requirements import parse_requirement_string


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
        requirements = []
        for req in d.get("requirements_file_requirements", []):
            parsed_req = parse_requirement_string(req)
            if parsed_req:
                requirements.append(parsed_req)
        return TrussSignature(
            content_hashes_by_path=d["content_hashes_by_path"],
            config=d["config"],
            requirements_file_requirements=requirements,
        )


ChangedPaths = Dict[str, List[str]]


@dataclass
class PatchDetails:
    prev_hash: str
    prev_signature: TrussSignature
    next_hash: str
    next_signature: TrussSignature
    patch_ops: List[Patch]

    def to_dict(self):
        return {
            "prev_hash": self.prev_hash,
            "prev_signature": self.prev_signature.to_dict(),
            "next_hash": self.next_hash,
            "next_signature": self.next_signature.to_dict(),
            "patch_ops": [patch_op.to_dict() for patch_op in self.patch_ops],
        }

    def is_empty(self) -> bool:
        # It's possible for prev_hash and next_hash to be different and yet
        # patch_ops to be empty, because certain parts of truss may be ignored.
        return len(self.patch_ops) == 0

    @staticmethod
    def from_dict(patch_details: Dict) -> "PatchDetails":
        return PatchDetails(
            prev_hash=patch_details["prev_hash"],
            prev_signature=TrussSignature.from_dict(patch_details["prev_signature"]),
            next_hash=patch_details["next_hash"],
            next_signature=TrussSignature.from_dict(patch_details["next_signature"]),
            patch_ops=[
                Patch.from_dict(patch_op) for patch_op in patch_details["patch_ops"]
            ],
        )


class PatchRequest(BaseModel):
    """Request to patch running container"""

    hash: str
    prev_hash: str
    patches: List[Patch]

    def to_dict(self):
        return {
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "patches": [patch.to_dict() for patch in self.patches],
        }

    @staticmethod
    def from_dict(patch_request_dict: Dict):
        current_hash = patch_request_dict["hash"]
        prev_hash = patch_request_dict["prev_hash"]
        patches = patch_request_dict["patches"]
        return PatchRequest(hash=current_hash, prev_hash=prev_hash, patches=patches)
