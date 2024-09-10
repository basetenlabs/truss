import boto3
from boto3.s3.transfer import TransferConfig
import os
import threading
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

MULTIPART_THRESHOLD = 50 * 1024 * 1024
MAX_CONCURRENCY = 10
CHUNK_SIZE = 8 * 1024 * 1024
MULTI_THREAD_WORKERS = 10

config = TransferConfig(
    multipart_threshold=MULTIPART_THRESHOLD,
    max_concurrency=MAX_CONCURRENCY,
    multipart_chunksize=CHUNK_SIZE,
    use_threads=True
)

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = 0
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def set_size(self, size):
        self._size = size

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            if self._size:
                percentage = (self._seen_so_far / self._size) * 100
                sys.stdout.write(
                    f"\r{self._filename}  {self._seen_so_far / (1024 * 1024):.2f}MB / {self._size / (1024 * 1024):.2f}MB  ({percentage:.2f}%)")
            else:
                sys.stdout.write(f"\r{self._filename}  {self._seen_so_far / (1024 * 1024):.2f}MB")
            sys.stdout.flush()

def download_file(s3, bucket_name, key, local_path):
    print(f"Downloading {key} to {local_path}...")
    
    progress = ProgressPercentage(local_path)
    
    response = s3.head_object(Bucket=bucket_name, Key=key)
    progress.set_size(response['ContentLength'])
    
    try:
        s3.download_file(
            bucket_name, 
            key, 
            local_path, 
            Config=config,
            Callback=progress
        )
        print(f"\nDownloaded {key} to {local_path}")
    except Exception as e:
        print(f"Error downloading {key} to {local_path}: {e}")
        raise

def download_model_directory(s3, bucket_name, prefix, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    file_keys = []

    for page in pages:
        if 'Contents' not in page:
            print(f"No files found in {prefix}")
            return

        for obj in page['Contents']:
            file_keys.append(obj['Key'])
    
    with ThreadPoolExecutor(max_workers=MULTI_THREAD_WORKERS) as executor:
        print(f"Downloading {len(file_keys)} files from {bucket_name}/{prefix} to {local_dir}...")
        future_to_key = {}
        for key in file_keys:
            relative_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            future = executor.submit(download_file, s3, bucket_name, key, local_path)
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {key}: {e}")

def download_model(bucket_name, model_key, local_path, aws_region=None):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not aws_access_key_id or not aws_secret_access_key:
        raise EnvironmentError('AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables.')

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    s3 = session.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=model_key)
        download_file(s3, bucket_name, model_key, local_path)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"{model_key} is a directory, downloading recursively.")
            download_model_directory(s3, bucket_name, model_key, local_path)
        else:
            print(f"Error: {e}")
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download a model from S3.')
    
    parser.add_argument('--bucket_name', required=True, help='S3 bucket name')
    parser.add_argument('--model_key', required=True, help='S3 object key or directory prefix (path to the model in the bucket)')
    parser.add_argument('--local_path', required=True, help='Local file path or directory to save the model')
    parser.add_argument('--aws_region', default='us-east-1', help='AWS region (default: us-east-1)')

    args = parser.parse_args()

    download_model(
        args.bucket_name, 
        args.model_key, 
        args.local_path,
        aws_region=args.aws_region
    )
