def model_supports_predict_proba(model: object) -> bool:
    if not hasattr(model, "predict_proba"):
        return False
    if hasattr(
        model, "_check_proba"
    ):  # noqa eg Support Vector Machines *can* predict proba if they made certain choices while training
        try:
            model._check_proba()
            return True
        except AttributeError:
            return False
    return True


def assign_request_to_inputs_instances_after_validation(body: dict) -> dict:
    # we will treat "instances" and "inputs" the same
    if "instances" in body and "inputs" not in body:
        body["inputs"] = body["instances"]
    elif "inputs" in body and "instances" not in body:
        body["instances"] = body["inputs"]
    return body
