import boto3
from io import BytesIO

s3 = boto3.client("s3")

def get_byte_stream_for_s3_path(s3_path: str) -> BytesIO:
    """
    Load data directly into memory from S3 and return a byte stream.

    Parameters:
        s3_path (str): The S3 path to the file.

    Returns:
        BytesIO: A byte stream of the loaded data.
    """
    bucket_name, s3_key = s3_path.replace("s3://", "").split("/", 1)
    s3_object = s3.get_object(Bucket=bucket_name, Key=s3_key)
    return BytesIO(s3_object["Body"].read())

def download_from_s3(s3_path: str, local_path: str) -> None:
    """
    Download a file from S3 to a specified local path.

    Parameters:
        s3_path (str): The S3 path to the file.
        local_path (str): The local path where the file should be saved.
    """
    bucket_name, s3_key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket_name, s3_key, local_path)
