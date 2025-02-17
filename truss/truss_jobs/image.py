import datetime
import pathlib
import uuid
from typing import TYPE_CHECKING, Optional, Type, Union

from truss.base.images import BasetenImage, CustomImage, ImageSpec
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3

if TYPE_CHECKING:
    from rich import progress

"""

class DockerAuthV1(BaseRestAPIModel):
    auth_method: DockerAuthType = Field(description="The method to authenticate with")
    secret_name: str = Field(description="The name of the secret to use for authentication. This secret must be defined on the baseten workspace.")

class CustomImageRequestV1(BaseRestAPIModel):
    image: str = Field(description="The image to use for the job")
    docker_auth: Optional[DockerAuthV1] = Field(default=None, description="The auth specification for the docker image")

class DockerImageRequestV1(BaseRestAPIModel):
    base_image: CustomImageRequestV1 = Field(description="The base image to use for the job")
    apt_requirements: list[str] = Field(description="A list of apt requirements to install")
    pip_requirements: list[str] = Field(description="A list of pip requirements to install")
    s3_key: Optional[str] = Field(default=None, description="The S3 key of bundled assets to include in the image")

class BuildImageRequestV1(BaseRestAPIModel):
    image_name: str = Field(description="The name of the image to build")
    image_tag: str = Field(description="The tag of the image to build")
    docker_image_request: DockerImageRequestV1 = Field(description="The request to build the image")
    build_secrets: list[str] = Field(default=[], description="A list of secrets to include in the build")

class ImageV1(BaseRestAPIModel):
    image_name: str = Field(description="The name of the image")
    image_tag: str = Field(description="The tag of the image")
    image_id: str = Field(description="The ID of the image")

"""


def build_image_request(
    api: BasetenApi,
    organization_id: str,
    image_spec: ImageSpec,
    progress_bar: Optional[Type["progress.Progress"]] = None,
) -> dict:
    """Build a docker image request from an image spec. It should be in the format of a BuildImageRequestV1, but should be
    a JSON request.
    """
    # generate an image tag if none exists
    image_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if image_spec.image_tag is not None:
        image_tag = image_spec.image_tag
    # TODO: we should clear out bundles for the supplied image tag
    # upload assets to s3
    file_bundles = []
    timestamp = datetime.datetime.now().isoformat()
    if image_spec.docker_image.file_bundles:
        for file_bundle in image_spec.docker_image.file_bundles:
            # upload bundle to s3
            temp_file = create_tar_with_progress_bar(
                pathlib.Path(file_bundle.source_path.abs_path),
                progress_bar=progress_bar,
            )
            # TODO: use organization ID
            s3_key = f"images/{image_spec.name}/{image_tag}/bundles/{timestamp}/{uuid.uuid4()}"
            temp_credentials_s3_upload = api.model_s3_upload_credentials()
            s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
            multipart_upload_boto3(
                temp_file.name,
                s3_bucket,
                s3_key,
                temp_credentials_s3_upload,
                progress_bar,
            )

            file_bundles.append(
                {"remote_path": file_bundle.remote_path, "s3_key": s3_key}
            )
    # get pip requirements from the file or from the listed pip requirements
    pip_requirements = []
    if image_spec.docker_image.pip_requirements_file is not None:
        with open(image_spec.docker_image.pip_requirements_file.abs_path, "r") as f:
            pip_requirements.extend(f.readlines())
    pip_requirements = image_spec.docker_image.pip_requirements

    docker_auth = None
    image_details: Union[str, dict] = ""
    if isinstance(image_spec.docker_image.base_image, CustomImage):
        docker_auth = None
        if image_spec.docker_image.base_image.docker_auth is not None:
            docker_auth = image_spec.docker_image.base_image.docker_auth.to_dict()
        image_details = {
            "image": image_spec.docker_image.base_image.image,
            "docker_auth": docker_auth,
        }
        print(image_details)
    elif isinstance(image_spec.docker_image.base_image, BasetenImage):
        image_details = image_spec.docker_image.base_image.value
    else:
        raise ValueError(
            f"Invalid base image type: {type(image_spec.docker_image.base_image)}"
        )

    return {
        "image_name": image_spec.name,
        "image_tag": image_tag,
        "docker_image_request": {
            "base_image": image_details,
            "apt_requirements": image_spec.docker_image.apt_requirements,
            "pip_requirements": pip_requirements,
            "file_bundles": file_bundles,
        },
        "build_secrets": image_spec.build_secrets,
    }
