from typing import Dict, List

from packaging import requirements  # type: ignore


def identify_requirement_name(req: str) -> str:
    try:
        parsed_req = requirements.Requirement(req)
        return parsed_req.name  # type: ignore
    except (requirements.InvalidRequirement, ValueError):
        # default to the whole line if we can't parse it.
        return req.strip()


def reqs_by_name(reqs: List[str]) -> Dict[str, str]:
    return {identify_requirement_name(req): req for req in reqs}
