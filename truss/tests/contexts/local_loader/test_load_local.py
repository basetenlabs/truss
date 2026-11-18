from truss.base.truss_spec import TrussSpec
from truss.truss_handle.truss_handle import TrussHandle


def test_add_secret_list(custom_model_truss_dir):
    handle = TrussHandle(custom_model_truss_dir)
    handle._update_config(secrets=["existing"])
    handle.add_secret("new")
    handle.add_secret("new")
    assert TrussSpec(custom_model_truss_dir).config.secrets == ["existing", "new"]
    # A default value can't be expressed in list form.
    handle.add_secret("with_value", "value")
    assert TrussSpec(custom_model_truss_dir).config.secrets == {
        "existing": None,
        "new": None,
        "with_value": "value",
    }
