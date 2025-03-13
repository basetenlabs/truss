from truss.base.truss_spec import TrussSpec
from truss.truss_handle.build import init_directory, load
from truss_chains.deployment import code_gen


def test_truss_init(tmp_path):
    spec = TrussSpec(init_directory(tmp_path))
    assert spec.model_module_dir.exists()
    assert spec.data_dir.exists()
    assert spec.truss_dir == tmp_path
    assert spec.config_path.exists()


def test_truss_init_with_python_dx(tmp_path):
    init_directory(tmp_path, model_name="Test Model Name", python_config=True)

    generated_truss_dir = code_gen.gen_truss_model_from_source(tmp_path / "my_model.py")
    truss_handle = load(generated_truss_dir)

    assert truss_handle.spec.config.model_name == "Test Model Name"
