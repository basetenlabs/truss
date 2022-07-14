import uuid
from typing import Dict

import boto3


def upload_file_to_s3(
    file_name, bucket: str = None, object_name: str = None, aws_credentials: Dict = None
) -> str:
    aws_secret_access_key = aws_credentials['aws_secret_access_key']
    aws_access_key_id = aws_credentials['aws_access_key_id']
    aws_region = aws_credentials['aws_region']
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    if object_name is None:
        object_name = f'{str(uuid.uuid4())}.png'
    s3.upload_file(file_name, bucket, object_name, ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'})
    url = f'https://{bucket}.s3.{aws_region}.amazonaws.com/{object_name}'
    return url
