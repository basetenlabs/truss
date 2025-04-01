import pathlib
from typing import List

from truss.base.custom_types import SafeModel
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import archive_dir
from truss.remote.baseten.utils import transfer
from truss_train.definitions import TrainingJob


class S3Artifact(SafeModel):
    s3_bucket: str
    s3_key: str


# Performs validation/transformation of fields that we don't want to expose
# to the end user via the TrainingJob SDK.
class PreparedTrainingJob(TrainingJob):
    runtime_artifacts: List[S3Artifact] = []

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # NB(nikhil): nest our processed artifacts back under runtime
        data["runtime"]["artifacts"] = data.pop("runtime_artifacts")
        return data


def prepare_push(api: BasetenApi, config: pathlib.Path, training_job: TrainingJob):
    # Assume config is at the root of the directory.
    archive = archive_dir(config.absolute().parent)
    credentials = api.get_blob_credentials(b10_types.BlobType.TRAIN)
    transfer.multipart_upload_boto3(
        archive.name,
        credentials["s3_bucket"],
        credentials["s3_key"],
        credentials["creds"],
    )

    return PreparedTrainingJob(
        image=training_job.image,
        runtime=training_job.runtime,
        compute=training_job.compute,
        runtime_artifacts=[
            S3Artifact(s3_key=credentials["s3_key"], s3_bucket=credentials["s3_bucket"])
        ],
    )
