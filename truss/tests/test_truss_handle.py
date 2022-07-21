import os
from pathlib import Path

import pytest

from truss.docker import Docker
from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.test_testing_utilities_for_other_tests import (
    ensure_kill_all, kill_all_with_retries)
from truss.truss_handle import TrussHandle
from truss.types import Example


def test_spec(custom_model_truss_dir_with_pre_and_post):
    dir_path = custom_model_truss_dir_with_pre_and_post
    th = TrussHandle(dir_path)
    spec = th.spec
    assert spec.truss_dir == dir_path


def test_description(custom_model_truss_dir_with_pre_and_post_description):
    dir_path = custom_model_truss_dir_with_pre_and_post_description
    th = TrussHandle(dir_path)
    spec = th.spec
    assert spec.description == "This model adds 3 to all inputs"


def test_server_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    resp = th.server_predict({
        'inputs': [1, 2, 3, 4],
    })
    assert resp == {'predictions': [4, 5, 6, 7]}


def test_readme_generation_int_example(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace('\n', '')
    correct_readme_contents = _read_readme('readme_int_example.md')
    assert readme_contents == correct_readme_contents


def test_readme_generation_no_example(custom_model_truss_dir_with_pre_and_post_no_example):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_no_example)
    # Remove the examples file
    os.remove(th._spec.examples_path)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace('\n', '')
    correct_readme_contents = _read_readme('readme_no_example.md')
    assert readme_contents == correct_readme_contents


def test_readme_generation_str_example(custom_model_truss_dir_with_pre_and_post_str_example):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_str_example)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace('\n', '')
    correct_readme_contents = _read_readme('readme_str_example.md')
    assert readme_contents == correct_readme_contents


@pytest.mark.integration
def test_build_docker_image(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = 'test-build-image-tag:0.0.1'
    image = th.build_docker_image(tag=tag)
    assert image.repo_tags[0] == tag


@pytest.mark.integration
def test_build_docker_image_gpu(custom_model_truss_dir_for_gpu, tmp_path):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = 'test-build-image-gpu-tag:0.0.1'
    build_dir = tmp_path / 'scaffold_build_dir'
    image = th.build_docker_image(tag=tag, build_dir=build_dir)
    assert image.repo_tags[0] == tag


@pytest.mark.integration
def test_docker_run(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = 'test-docker-run-tag:0.0.1'
    container = th.docker_run(tag=tag)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.skip(reason='Needs gpu')
@pytest.mark.integration
def test_docker_run_gpu(custom_model_truss_dir_for_gpu):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = 'test-docker-run-gpu-tag:0.0.1'
    container = th.docker_run(tag=tag)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_docker_run_without_tag(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    container = th.docker_run()
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def get_docker_containers_from_labels(
    custom_model_truss_dir_with_pre_and_post
):
    with ensure_kill_all():
        t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
        assert(len(t1.get_docker_containers_from_labels()) == 0)
        t1.docker_run()
        assert(len(t1.get_docker_containers_from_labels()) == 1)
        t1.docker_run(port=3000)
        assert(len(t1.get_docker_containers_from_labels()) == 2)
        t1.kill_container()
        assert(len(t1.get_docker_containers_from_labels()) == 0)


@pytest.mark.integration
def test_docker_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = 'test-docker-predict-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1, 2]}, tag=tag)
        assert result == {'predictions': [4, 5]}


@pytest.mark.integration
def test_docker_multiple_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = 'test-docker-predict-tag:0.0.1'
    with ensure_kill_all():
        r1 = th.docker_predict({'inputs': [1, 2]}, tag=tag)
        r2 = th.docker_predict({'inputs': [3, 4]}, tag=tag)
        assert r1 == {'predictions': [4, 5]}
        assert r2 == {'predictions': [6, 7]}
        assert(len(th.get_docker_containers_from_labels()) == 1)


@pytest.mark.integration
def test_kill_all(custom_model_truss_dir, custom_model_truss_dir_with_pre_and_post):
    t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    t2 = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        t1.docker_run()
        assert(len(t1.get_docker_containers_from_labels()) == 1)
        t2.docker_run(local_port=3000)
        assert(len(t2.get_docker_containers_from_labels()) == 1)
        kill_all_with_retries()
        assert(len(t1.get_docker_containers_from_labels()) == 0)
        assert(len(t2.get_docker_containers_from_labels()) == 0)


@pytest.mark.skip(reason='Needs gpu')
@pytest.mark.integration
def test_docker_predict_gpu(custom_model_truss_dir_for_gpu):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = 'test-docker-predict-gpu-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1]}, tag=tag)
        assert result['predictions'][0]['cuda_version'].startswith('11')


@pytest.mark.integration
def test_docker_predict_secrets(custom_model_truss_dir_for_secrets):
    th = TrussHandle(custom_model_truss_dir_for_secrets)
    tag = 'test-docker-predict-secrets-tag:0.0.1'
    LocalConfigHandler.set_secret('secret_name', 'secret_value')
    with ensure_kill_all():
        try:
            result = th.docker_predict({'inputs': ['secret_name']}, tag=tag)
            assert result['predictions'][0] == 'secret_value'
        finally:
            LocalConfigHandler.remove_secret('secret_name')


@pytest.mark.integration
def test_docker_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    tag = 'test-docker-no-preprocess-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1]}, tag=tag)
        assert result['predictions'][0] == 2


@pytest.mark.integration
def test_local_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    result = th.server_predict({'inputs': [1]})
    assert result['predictions'][0] == 2


@pytest.mark.integration
def test_docker_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    tag = 'test-docker-no-postprocess-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1]}, tag=tag)
        assert result['predictions'][0] == 2


@pytest.mark.integration
def test_local_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    result = th.server_predict({'inputs': [1]})
    assert result['predictions'][0] == 2


@pytest.mark.integration
def test_docker_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    tag = 'test-docker-no-load-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1]}, tag=tag)
        assert result['predictions'][0] == 1


@pytest.mark.integration
def test_local_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    result = th.server_predict({'inputs': [1]})
    assert result['predictions'][0] == 1


@pytest.mark.integration
def test_docker_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    tag = 'test-docker-no-params-init-tag:0.0.1'
    with ensure_kill_all():
        result = th.docker_predict({'inputs': [1]}, tag=tag)
        assert result['predictions'][0] == 1


@pytest.mark.integration
def test_local_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    result = th.server_predict({'inputs': [1]})
    assert result['predictions'][0] == 1


@pytest.mark.integration
def test_custom_python_requirement(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_python_requirement('theano')
    th.add_python_requirement('scipy')
    tag = 'test-custom-python-req-tag:0.0.1'
    container = th.docker_run(tag=tag)
    try:
        _verify_python_requirement_installed_on_container(tag, 'theano')
        _verify_python_requirement_installed_on_container(tag, 'scipy')
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_custom_system_package(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_system_package('jq')
    th.add_system_package('fzf')
    tag = 'test-custom-system-package-tag:0.0.1'
    container = th.docker_run(tag=tag)
    try:
        _verify_system_package_installed_on_container(tag, 'jq')
        _verify_system_package_installed_on_container(tag, 'fzf')
    finally:
        Docker.client().kill(container)


def test_enable_gpu(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.enable_gpu()
    assert th.spec.config.resources.use_gpu


def test_update_requirements(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    requirements = [
        'tensorflow==2.3.1',
        'uvicorn==0.12.2',
    ]
    th.update_requirements(requirements)
    sc_requirements = th.spec.requirements
    assert sc_requirements == requirements


def test_update_requirements_from_file(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    requirements = [
        'tensorflow==2.3.1',
        'uvicorn==0.12.2',
    ]
    req_file_path = tmp_path / 'requirements.txt'
    with req_file_path.open('w') as req_file:
        for req in requirements:
            req_file.write(f'{req}\n')
    th.update_requirements_from_file(str(req_file_path))
    sc_requirements = th.spec.requirements
    assert sc_requirements == requirements


@pytest.mark.integration
def test_add_environment_variable(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_environment_variable('test_env', 'test_value')
    tag = 'test-add-env-var-tag:0.0.1'
    container = th.docker_run(tag=tag)
    try:
        _verify_environment_variable_on_container(tag, 'test_env', 'test_value')
    finally:
        Docker.client().kill(container)


def test_add_data_file(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    data_filepath = tmp_path / 'test_data.txt'
    with data_filepath.open('w') as data_file:
        data_file.write('test')

    th.add_data(str(data_filepath))

    scaf_data_filepath = th.spec.data_dir / 'test_data.txt'
    assert scaf_data_filepath.exists()
    with scaf_data_filepath.open() as data_file:
        assert data_file.read() == 'test'


def test_add_data_fileglob(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    file1_path = tmp_path / 'test_data1.txt'
    with file1_path.open('w') as data_file:
        data_file.write('test1')

    file2_path = tmp_path / 'test_data2.txt'
    with file2_path.open('w') as data_file:
        data_file.write('test2')

    file2_path = tmp_path / 'test_data3.json'
    with file2_path.open('w') as data_file:
        data_file.write('{}')

    th.add_data(f'{str(tmp_path)}/*.txt')

    assert (th.spec.data_dir / 'test_data1.txt').exists()
    assert (th.spec.data_dir / 'test_data2.txt').exists()
    assert not (th.spec.data_dir / 'test_data2.json').exists()


def test_add_data_dir(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    sub_dir = tmp_path / 'sub'
    sub_sub_dir = sub_dir / 'sub'
    sub_sub_dir.mkdir(parents=True)

    file_path = sub_sub_dir / 'test_file.txt'
    with file_path.open('w') as data_file:
        data_file.write('test')

    th.add_data(str(sub_dir))

    scaf_file_path = th.spec.data_dir / 'sub' / 'sub' / 'test_file.txt'
    assert scaf_file_path.exists()
    with scaf_file_path.open() as data_file:
        assert data_file.read() == 'test'


def test_examples(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    examples = th.examples()
    assert 'example1' in [example.name for example in examples]


def test_example(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    assert 'inputs' in th.example('example1').input


def test_example_index(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    assert 'inputs' in th.example(0).input


def test_add_example_new(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    orig_examples = th.examples()
    th.add_example('example2', {'inputs': [[1]]})
    assert th.examples() == [
        *orig_examples,
        Example('example2', {'inputs': [[1]]}),
    ]


def test_add_example_update(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_example('example1', {'inputs': [[1]]})
    assert th.examples() == [
        Example('example1', {'inputs': [[1]]}),
    ]


def test_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    resp = th.server_predict({
        'inputs': [1, 2, 3, 4],
    })
    assert resp == [1, 1, 1, 1]


@pytest.mark.integration
def test_docker_predict_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        resp = th.docker_predict({
            'inputs': [1, 2, 3, 4],
        })
        assert resp == [1, 1, 1, 1]


def _container_exists(container) -> bool:
    for row in Docker.client().ps():
        if row.id.startswith(container.id):
            return True
    return False


def _verify_system_package_installed_on_container(tag: str, cmd: str):
    resp = Docker.client().run(tag, ['which', cmd])
    assert resp.strip() == f'/usr/bin/{cmd}'


def _verify_python_requirement_installed_on_container(tag: str, cmd: str):
    resp = Docker.client().run(tag, ['pip', 'show', cmd])
    assert resp.splitlines()[0].lower() == f'Name: {cmd}'.lower()


def _verify_environment_variable_on_container(
    tag: str,
    env_var_name: str,
    env_var_value: str,
):
    resp = Docker.client().run(tag, ['env'])
    needle = f'{env_var_name}={env_var_value}'
    assert needle in resp.splitlines()


def _read_readme(filename: str) -> str:
    readme_correct_path = Path(__file__).parent.parent / 'test_data' / filename
    readme_contents = readme_correct_path.open().read().replace('\n', '')
    return readme_contents
