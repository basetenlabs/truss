import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from truss.build import cleanup, init, mk_truss
from truss.truss_spec import TrussSpec


def test_truss_init(tmp_path):
    dir_name = str(tmp_path)
    init(dir_name)
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.data_dir.exists()
    assert spec.truss_dir == tmp_path
    assert spec.config_path.exists()


def test_truss_init_with_data_file_and_requirements_file_and_bundled_packages(
    tmp_path,
):
    dir_path = tmp_path / "truss"
    dir_name = str(dir_path)

    # Init data files
    data_path = tmp_path / "data.txt"
    with data_path.open("w") as data_file:
        data_file.write("test")

    # Init requirements file
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")

    # init bundled packages
    packages_path = tmp_path / "dep_pkg"
    packages_path.mkdir()
    packages_path_file_py = packages_path / "file.py"
    packages_path_init_py = packages_path / "__init__.py"
    pkg_files = [packages_path_init_py, packages_path_file_py]
    for pkg_file in pkg_files:
        with pkg_file.open("w") as fh:
            fh.write("test")

    init(
        dir_name,
        data_files=[str(data_path)],
        requirements_file=str(req_file_path),
        bundled_packages=[str(packages_path)],
    )
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert spec.bundled_packages_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements
    assert (spec.bundled_packages_dir / "dep_pkg" / "__init__.py").exists()
    assert (spec.bundled_packages_dir / "dep_pkg" / "file.py").exists()


def test_mk_truss(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / "truss"
    data_file_path = tmp_path / "data.txt"
    with data_file_path.open("w") as data_file:
        data_file.write("test")
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")
    scaf = mk_truss(
        sklearn_rfc_model,
        target_directory=dir_path,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    spec = scaf.spec
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements


def test_mk_truss_pipeline(sklearn_rfc_model, tmp_path):
    def inference(request: dict):
        inputs = request["inputs"]
        response = sklearn_rfc_model.predict([inputs])[0]
        return {"result": response}

    dir_path = tmp_path / "truss"
    data_file_path = tmp_path / "data.txt"
    with data_file_path.open("w") as data_file:
        data_file.write("test")
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")
    scaf = mk_truss(
        inference,
        target_directory=dir_path,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    spec = scaf.spec
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements


def test_truss_sklearn_predict(sklearn_rfc_model):
    with _model_server_predict(sklearn_rfc_model, {"inputs": [[0, 0, 0, 0]]}) as result:
        assert "predictions" in result
        assert "probabilities" in result
        probabilities = result["probabilities"]
        assert np.shape(probabilities) == (1, 3)


def test_truss_sklearn_predict_pipeline(sklearn_rfc_model):
    def inference(request: dict):
        inputs = request["inputs"]
        response = sklearn_rfc_model.predict([inputs])[0]
        return {"result": response}

    with _model_server_predict_pipeline(inference, {"inputs": [0, 0, 0, 0]}) as result:
        assert "result" in result
        assert result["result"] == 0


def test_truss_keras_predict(keras_mpg_model):
    with _model_server_predict(
        keras_mpg_model,
        {"inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ) as result:
        assert "predictions" in result
        predictions = result["predictions"]
        assert np.shape(predictions) == (1, 1)


def test_truss_keras_predict_pipeline(keras_mpg_model):
    def inference(request: dict):
        inputs = request["inputs"]
        response = keras_mpg_model.predict(inputs)
        return {"result": response}

    with _model_server_predict_pipeline(
        inference,
        {"inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ) as result:
        assert "result" in result
        predictions = result["result"]
        assert np.shape(predictions) == (1, 1)


def test_truss_pytorch_predict(pytorch_model):
    model = pytorch_model[0]
    with _model_server_predict(
        model,
        {"inputs": [[0, 0, 0]]},
    ) as result:
        assert "predictions" in result
        assert len(result["predictions"]) == 1


def test_truss_huggingface_transformer_predict(
    huggingface_transformer_t5_small_pipeline,
):
    with _model_server_predict(
        huggingface_transformer_t5_small_pipeline,
        {"inputs": ["My name is Sarah and I live in London"]},
    ) as result:
        print(result)
        assert "predictions" in result
        predictions = result["predictions"]
        assert len(predictions) == 1
        assert predictions[0]["generated_text"].startswith("Mein Name")


def test_cleanup(sklearn_rfc_model, tmp_path):
    data_file_path = tmp_path / "data.txt"
    with data_file_path.open("w") as data_file:
        data_file.write("test")
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")
    _ = mk_truss(
        sklearn_rfc_model,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    cleanup()
    build_folder_path = Path(Path.home(), ".truss")
    directory = list(build_folder_path.glob("**/*"))
    files = [obj.name for obj in directory if obj.is_file()]
    unique_files = set(files)
    assert build_folder_path.exists()
    assert unique_files == {"config.yaml"}


def test_truss_via_simple_mk_pipeline():
    def generate(request):
        x = request["x"]
        return {"y": x + 1}

    with _model_server_predict_pipeline(generate, {"x": 1}) as result:
        assert result["y"] == 2


def test_truss_via_t5_mk_pipeline(
    huggingface_transformer_t5_small_model, huggingface_transformer_t5_small_tokenizer
):
    def generate(request):
        prompt = request["prompt"]
        input_ids = huggingface_transformer_t5_small_tokenizer(
            prompt, return_tensors="pt"
        ).input_ids
        return {
            "response": huggingface_transformer_t5_small_tokenizer.decode(
                huggingface_transformer_t5_small_model.generate(input_ids)[0],
                skip_special_tokens=True,
            )
        }

    with _model_server_predict_pipeline(
        generate, {"prompt": "translate to french: hello"}
    ) as result:
        assert result["response"].startswith("Hallo")


@contextmanager
def _model_server_predict(model, model_input):
    with tempfile.TemporaryDirectory() as dir_name:
        sc = mk_truss(model, target_directory=dir_name)
        result = sc.server_predict(model_input)
        yield result


@contextmanager
def _model_server_predict_pipeline(pipeline, model_input):
    sc = mk_truss(pipeline)
    result = sc.server_predict(model_input)
    yield result
