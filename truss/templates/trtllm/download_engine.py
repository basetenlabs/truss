import argparse
from huggingface_hub import snapshot_download


def download_engine(engine_repository: str, output_dir: str):
    snapshot_download(
        engine_repository,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        max_workers=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and setup engine from Hugging Face"
    )
    parser.add_argument(
        "--engine-repository", required=True, type=str, help="Engine repository name"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory to store engine files",
    )

    args = parser.parse_args()

    download_engine(args.engine_repository, args.output_dir)
