import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from packaging import requirements  # type: ignore

URL_PATTERN = re.compile(r"^(https?|git|svn|hg|bzr)(\+|:\/\/)")


def is_url_based_requirement(req: str) -> bool:
    return bool(URL_PATTERN.match(req.strip()))


def get_egg_tag(req: str) -> Optional[List[str]]:
    if is_url_based_requirement(req):
        parsed_url = urlparse(req)
        fragments = parse_qs(parsed_url.fragment)
        return fragments.get("egg", None)
    return None


def reqs_by_name(reqs: List[str]) -> Dict[str, str]:
    return {identify_requirement_name(req): req for req in reqs if req.strip()}


def identify_requirement_name(req: str) -> str:
    try:
        # parse as a url requirement
        req = req.strip()
        if is_url_based_requirement(req):
            parsed_url = urlparse(req)
            # Identify the package by it's unique components. We can't reliably id a package name
            # via URL, so we fallback on this instead.
            vcs_options = ["git", "svn", "hg", "bzr"]
            for vcs in vcs_options:
                if req.startswith(vcs + "+"):
                    return f"{vcs}+{parsed_url.netloc}{parsed_url.path.split('@')[0]}"

            return req

        parsed_req = requirements.Requirement(req)
        return parsed_req.name  # type: ignore
    except (requirements.InvalidRequirement, ValueError):
        # default to the whole line if we can't parse it.
        return req.strip()


@dataclass
class RequirementMeta:
    requirement: str
    name: str
    is_url_based_requirement: bool
    egg_tag: Optional[List[str]]

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def from_req(req: str) -> "RequirementMeta":
        return RequirementMeta(
            req,
            identify_requirement_name(req),
            is_url_based_requirement(req),
            get_egg_tag(req),
        )
