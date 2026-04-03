import logging
from contextlib import contextmanager
from typing import Generator

from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@contextmanager
def handle_client_error(
    operation_description: str = "AWS operation",
) -> Generator[None, None, None]:
    """
    Context manager to handle common boto3 errors and convert them to RuntimeError.

    Args:
        operation_description: Description of the operation being performed for error messages

    Raises:
        RuntimeError: For NoCredentialsError, ClientError, and other exceptions
    """
    try:
        yield
    except NoCredentialsError as nce:
        raise RuntimeError(
            f"No AWS credentials found for {operation_description}\nOriginal exception: {str(nce)}"
        )
    except ClientError as ce:
        raise RuntimeError(
            f"AWS client error when {operation_description} (check your credentials): {str(ce)}"
        )

    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error `{exc}` during `{operation_description}`\nOriginal exception: {str(exc)}"
        )
