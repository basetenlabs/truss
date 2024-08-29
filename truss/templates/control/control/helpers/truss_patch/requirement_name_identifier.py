from typing import Dict, List
from urllib.parse import urlparse, parse_qs
import re
import logging

from packaging import requirements  # type: ignore
logger: logging.Logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"^(https?|git|svn|hg|bzr)\+")

def is_url_based_requirement(req: str) -> bool:
    return bool(URL_PATTERN.match(req))

def identify_requirement_name(req: str) -> str:
    try:
                # parse as a url requirement
        req = req.strip()
        if is_url_based_requirement(req):
            parsed_url = urlparse(req)
            fragments = parse_qs(parsed_url.fragment)
            if 'egg' not in fragments:
                logger.warning(f'Url-based requirement "{req}" is missing egg tag. Removal during `truss watch` will be ignored')

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


def reqs_by_name(reqs: List[str]) -> Dict[str, str]:
    return {identify_requirement_name(req): req for req in reqs if req.strip()}
