def model_supports_predict_proba(model):
    if not hasattr(model, 'predict_proba'):
        return False
    if hasattr(model, '_check_proba'):  # noqa eg Support Vector Machines *can* predict proba if they made certain choices while training
        try:
            model._check_proba()
            return True
        except AttributeError:
            return False
    return True


def assign_request_to_inputs_instances_after_validation(request):
    # we will treat "instances" and "inputs" the same
    if "instances" in request and "inputs" not in request:
        request["inputs"] = request["instances"]
    elif "inputs" in request and "instances" not in request:
        request["instances"] = request["inputs"]
    return request
