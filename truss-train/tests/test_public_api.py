import pathlib
import tempfile
from unittest import mock

from truss.remote.remote_factory import API_KEY_ENV, ENV_REMOTE_NAME, REMOTE_URL_ENV
from truss_train.definitions import (
    Compute,
    Image,
    Runtime,
    TrainingJob,
    TrainingProject,
)
from truss_train.public_api import push


def _make_job_resp(project_id="proj-123", job_id="job-456", project_name="my-project"):
    return {"id": job_id, "project_id": project_id, "project_name": project_name}


class TestPushWithTrainingProject:
    @mock.patch("truss_train.public_api._upsert_project_and_create_job")
    @mock.patch("truss.remote.remote_factory.RemoteFactory.create")
    def test_push_training_project(self, mock_create_remote, mock_upsert):
        mock_upsert.return_value = _make_job_resp()
        mock_remote = mock.Mock()
        mock_create_remote.return_value = mock_remote

        project = TrainingProject(
            name="my-project",
            job=TrainingJob(
                image=Image(base_image="hello-world"),
                compute=Compute(),
                runtime=Runtime(),
            ),
        )

        result = push(project, remote="baseten")

        assert result == _make_job_resp()

        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        assert call_args[0][0] is mock_remote
        assert call_args[0][1] is project
        # source_dir defaults to cwd
        assert call_args[0][2] == pathlib.Path.cwd()

    @mock.patch("truss_train.public_api._upsert_project_and_create_job")
    @mock.patch("truss.remote.remote_factory.RemoteFactory.create")
    def test_push_training_project_with_source_dir(
        self, mock_create_remote, mock_upsert, tmp_path
    ):
        mock_upsert.return_value = _make_job_resp()
        mock_create_remote.return_value = mock.Mock()

        project = TrainingProject(
            name="my-project",
            job=TrainingJob(
                image=Image(base_image="hello-world"),
                compute=Compute(),
                runtime=Runtime(),
            ),
        )

        result = push(project, source_dir=tmp_path)

        assert result == _make_job_resp()
        call_args = mock_upsert.call_args
        assert call_args[0][2] == tmp_path


class TestPushWithPath:
    @mock.patch("truss_train.public_api.create_training_job")
    @mock.patch("truss.remote.remote_factory.RemoteFactory.create")
    def test_push_config_path(self, mock_create_remote, mock_create_job):
        mock_create_job.return_value = _make_job_resp()
        mock_remote = mock.Mock()
        mock_create_remote.return_value = mock_remote

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "from truss_train.definitions import TrainingProject, TrainingJob, Image, Compute, Runtime\n"
                "project = TrainingProject(\n"
                '    name="file-project",\n'
                "    job=TrainingJob(\n"
                '        image=Image(base_image="hello-world"),\n'
                "        compute=Compute(),\n"
                "        runtime=Runtime(),\n"
                "    ),\n"
                ")\n"
            )
            config_path = pathlib.Path(f.name)

        result = push(config_path, remote="baseten")

        assert result == _make_job_resp()

        mock_create_job.assert_called_once()
        call_args = mock_create_job.call_args
        assert call_args[0][0] is mock_remote
        assert call_args[0][1] == config_path

        config_path.unlink()


def _project() -> TrainingProject:
    return TrainingProject(
        name="my-project",
        job=TrainingJob(
            image=Image(base_image="hello-world"), compute=Compute(), runtime=Runtime()
        ),
    )


@mock.patch("truss_train.public_api._upsert_project_and_create_job")
@mock.patch("truss.remote.remote_factory.RemoteFactory.create")
def test_push_without_a_remote_uses_the_env_remote(
    mock_create_remote, mock_upsert, monkeypatch
):
    monkeypatch.setenv(REMOTE_URL_ENV, "https://app.test.com")
    monkeypatch.setenv(API_KEY_ENV, "env_key")

    push(_project())

    mock_create_remote.assert_called_once_with(remote=ENV_REMOTE_NAME)


@mock.patch("truss_train.public_api._upsert_project_and_create_job")
@mock.patch("truss.remote.remote_factory.RemoteFactory.create")
def test_push_without_a_remote_defaults_to_baseten(
    mock_create_remote, mock_upsert, monkeypatch
):
    monkeypatch.delenv(REMOTE_URL_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    push(_project())

    mock_create_remote.assert_called_once_with(remote="baseten")
