STATUS_TO_DISPLAYABLE = {
    "BUILDING_MODEL": "BUILDING",
    "DEPLOYING_MODEL": "DEPLOYING",
    "MODEL_DEPLOY_FAILED": "DEPLOY_FAILED",
    "MODEL_LOADING": "LOADING_MODEL",
    "MODEL_READY": "ACTIVE",
    "MODEL_UNHEALTHY": "UNHEALTHY",
    "BUILDING_MODEL_FAILED": "BUILD_FAILED",
    "BUILDING_MODEL_STOPPED": "BUILD_STOPPED",
    "DEACTIVATING_MODEL": "DEACTIVATING",
    "DEACTIVATED_MODEL": "INACTIVE",
    "MODEL_DNE_ERROR": "FAILED",
    "UPDATING": "UPDATING",
    "MIGRATING_WORKLOAD_PLANES": "UPDATING",
    "SCALED_TO_ZERO": "SCALED_TO_ZERO",
    "SCALING_FROM_ZERO": "WAKING_UP",
}

# NB(nikhil): These are slightly translated verisons of our internal model state machine.
MODEL_RUNNING_STATES = [
    "BUILDING",
    "DEPLOYING",
    "LOADING_MODEL",
    "ACTIVE",
    "UPDATING",
    "WAKING_UP",
]


def get_displayable_status(status: str) -> str:
    """
    TODO: Remove this method once Chains is supported in the REST API

    This is used by the `truss chains deploy` command right now to
    print the right status. Once Chains are supported by the REST API, the
    Baseten REST API will return status strings matching the ones here, so we don't
    need to do any mapping.
    """
    return STATUS_TO_DISPLAYABLE.get(status, "UNKNOWN")
