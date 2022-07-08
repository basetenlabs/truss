import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from truss.build import cleanup, init, mk_truss
from truss.truss_spec import TrussSpec


def test_scaffold_init(tmp_path):
    dir_name = str(tmp_path)
    init(dir_name)
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.data_dir.exists()
    assert spec.truss_dir == tmp_path
    assert spec.config_path.exists()


def test_scaffold_init_with_data_file_and_requirements_file(tmp_path):
    dir_path = tmp_path / 'scaffold'
    dir_name = str(dir_path)
    data_file_path = tmp_path / 'data.txt'
    with data_file_path.open('w') as data_file:
        data_file.write('test')
    req_file_path = tmp_path / 'requirements.txt'
    requirements = [
        'tensorflow==2.3.1',
        'uvicorn==0.12.2',
    ]
    with req_file_path.open('w') as req_file:
        for req in requirements:
            req_file.write(f'{req}\n')

    init(dir_name, data_files=[str(data_file_path)], requirements_file=str(req_file_path))
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert (spec.data_dir / 'data.txt').exists()
    assert spec.requirements == requirements


def test_scaffold(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / 'scaffold'
    data_file_path = tmp_path / 'data.txt'
    with data_file_path.open('w') as data_file:
        data_file.write('test')
    req_file_path = tmp_path / 'requirements.txt'
    requirements = [
        'tensorflow==2.3.1',
        'uvicorn==0.12.2',
    ]
    with req_file_path.open('w') as req_file:
        for req in requirements:
            req_file.write(f'{req}\n')
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
    assert (spec.data_dir / 'data.txt').exists()
    assert spec.requirements == requirements


def test_scaffold_sklearn_predict(sklearn_rfc_model):
    with _model_server_predict(sklearn_rfc_model, {'inputs': [[0, 0, 0, 0]]}) as result:
        assert 'predictions' in result
        assert 'probabilities' in result
        probabilities = result['probabilities']
        assert np.shape(probabilities) == (1, 3)


def test_scaffold_keras_predict(keras_mpg_model):
    with _model_server_predict(
        keras_mpg_model,
        {'inputs': [[0, 0, 0, 0, 0, 0, 0, 0, 0]]},
    ) as result:
        assert 'predictions' in result
        predictions = result['predictions']
        assert np.shape(predictions) == (1, 1)


def test_scaffold_pytorch_predict(pytorch_model):
    model = pytorch_model[0]
    with _model_server_predict(
        model,
        {'inputs': [[0, 0, 0]]},
    ) as result:
        assert 'predictions' in result
        assert len(result['predictions']) == 1


def test_scaffold_huggingface_transformer_predict(huggingface_transformer_t5_small_model):
    with _model_server_predict(
        huggingface_transformer_t5_small_model,
        {'inputs': ['My name is Sarah and I live in London']},
    ) as result:
        print(result)
        assert 'predictions' in result
        predictions = result['predictions']
        assert len(predictions) == 1
        assert predictions[0]['generated_text'].startswith('Mein Name')


def test_cleanup(sklearn_rfc_model, tmp_path):
    data_file_path = tmp_path / 'data.txt'
    with data_file_path.open('w') as data_file:
        data_file.write('test')
    req_file_path = tmp_path / 'requirements.txt'
    requirements = [
        'tensorflow==2.3.1',
        'uvicorn==0.12.2',
    ]
    with req_file_path.open('w') as req_file:
        for req in requirements:
            req_file.write(f'{req}\n')
    _ = mk_truss(
        sklearn_rfc_model,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    cleanup()
    build_folder_path = Path(
        Path.home(),
        '.truss'
    )
    directory = list(build_folder_path.glob("**/*"))
    files = [obj.name for obj in directory if obj.is_file()]
    unique_files = set(files)
    assert build_folder_path.exists()
    assert unique_files == {'config.yaml'}


@contextmanager
def _model_server_predict(model, model_input):
    with tempfile.TemporaryDirectory() as dir_name:
        sc = mk_truss(model, target_directory=dir_name)
        result = sc.server_predict(model_input)
        yield result
