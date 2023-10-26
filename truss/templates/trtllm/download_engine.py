import argparse
from pathlib import Path
import shutil
import subprocess
import os


def download_engine(engine_repository: str, output_dir: str):
    # Identify existing files in the current directory
    previous_files = set(os.listdir(os.getcwd()))

    # Run subprocess to download the engine snapshot via git clone
    subprocess.run(
        ["git", "clone", f"https://huggingface.co/{engine_repository}"],
        capture_output=True,
    )

    # Identify new files in the current directory
    current_files = set(os.listdir(os.getcwd()))

    # Find the directory that was created by git clone
    engine_root = list(current_files - previous_files)[0]

    # Construct the full path of the newly created directory
    engine_root_path = Path(engine_root)

    # Search for the folder containing the .engine files
    engine_folder = None
    for root, _, files in os.walk(engine_root_path):
        if any(file.endswith(".engine") for file in files):
            engine_folder = Path(root)
            break

    # If we found the folder, move its contents to output_dir
    if engine_folder:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for item in engine_folder.iterdir():
            if item.is_file() or item.is_dir():
                shutil.move(str(item), output_path)
    else:
        print(f"No folder containing .engine files found under {engine_root_path}")


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
