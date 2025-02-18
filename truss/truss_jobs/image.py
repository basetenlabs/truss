import datetime
import pathlib
import uuid
from typing import TYPE_CHECKING, Optional, Type, Union

import truss
from truss.base.images import BasetenImage, CustomImage, ImageSpec
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.util.path import build_absolute_path, handle_path_or_str

if TYPE_CHECKING:
    from rich import progress


def build_image_request(
    cwd: pathlib.Path,
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
                build_absolute_path(handle_path_or_str(file_bundle.source_path)),
                progress_bar=progress_bar,
            )
            # TODO: use organization ID
            # TODO: need to do some validation on the image name and tag before hand
            # - ensure overwriting is intentional
            # - ensure the image name is valid
            s3_key = f"images/{image_spec.name}/{image_tag}/bundles/{timestamp}/{uuid.uuid4()}"
            temp_credentials_s3_upload = api.model_s3_upload_credentials()
            temp_credentials_s3_upload.pop("s3_key")
            s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
            multipart_upload_boto3(
                temp_file.name,
                s3_bucket,
                s3_key,
                temp_credentials_s3_upload,
                progress_bar,
            )

            file_bundles.append(
                {
                    "remote_path": build_absolute_path(
                        handle_path_or_str(file_bundle.remote_path)
                    ),
                    "s3_key": s3_key,
                }
            )
    # get pip requirements from the file or from the listed pip requirements
    pip_requirements = []
    if image_spec.docker_image.pip_requirements_file is not None:
        with open(
            build_absolute_path(
                handle_path_or_str(image_spec.docker_image.pip_requirements_file)
            ),
            "r",
        ) as f:
            pip_requirements.extend(f.readlines())
    else:
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

    # TODO: validate envvars that are secrets
    return {
        "image_name": image_spec.name,
        "image_tag": image_tag,
        "docker_image_request": {
            "base_image": image_details,
            "apt_requirements": image_spec.docker_image.apt_requirements,
            "pip_requirements": pip_requirements,
            "file_bundles": file_bundles,
        },
        "build_envvars": [envvar.dict() for envvar in image_spec.build_envvars],
        "build_commands": image_spec.build_commands,
        "truss_version": truss.version(),
    }
