import os
import enum
import shutil
import subprocess
import sys
import yaml
import json

from pathlib import Path
from typing import List


DOC_CONFIGURATION_FILE = "doc.yaml"
TRUSS_EXAMPLES_REPO = "https://github.com/basetenlabs/truss-examples-2"
DESTINATION_DIR = "truss-examples-2"
MINT_CONFIG_PATH = "docs/mint.json"


class FileType(enum.Enum):
    YAML = "yaml"
    PYTHON = "python"


def clone_repo():
    """
    If the destination directory exists, remove it.
    Then, clone the given repo into the specified directory.
    """
    if os.path.exists(DESTINATION_DIR):
        shutil.rmtree(DESTINATION_DIR)

    try:
        subprocess.run(
            ["git", "clone", TRUSS_EXAMPLES_REPO, DESTINATION_DIR], check=True
        )
        print(f"Successfully cloned {TRUSS_EXAMPLES_REPO} to {DESTINATION_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning the repo: {e}")
        sys.exit(1)


def fetch_file_contents(path: str):
    with open(path, "r") as f:
        return f.read()


def _fetch_example_dirs(root_dir):
    """
    Walk through the directory structure from the root directory and
    find all directories that have the specified file in it.
    """
    dirs_with_file = []

    for dirpath, _, filenames in os.walk(root_dir):
        if DOC_CONFIGURATION_FILE in filenames:
            dirs_with_file.append(dirpath)

    return dirs_with_file


def _get_example_destination(truss_directory) -> Path:
    """
    Get the destination directory for the example.
    """
    original_path = Path(truss_directory)
    folder, example = original_path.parts[1:]
    example_file = f"{example}.mdx"
    return Path("docs/examples") / folder / example_file


def _get_file_type(file_path: str) -> FileType:
    _, extension = os.path.splitext(file_path)
    if extension == ".yaml":
        return FileType.YAML

    if extension == ".py":
        return FileType.PYTHON

    raise ValueError(f"Unknown file type: {extension}")


class ContentBlock:
    def formatted_content(self):
        raise NotImplementedError


class CodeBlock(ContentBlock):
    def __init__(self, file_type: FileType, file_path: str):
        self.file_type = file_type
        self.file_path = file_path
        self.content = ""

    def formatted_content(self):
        return f"\n```{self.file_type.value} {self.file_path}\n{self.content}```"


class MarkdownBlock(ContentBlock):
    def __init__(self, content: str):
        self.content = content

    def formatted_content(self):
        # Remove the first comment and space character, such that
        # "# Hello" becomes "Hello
        return self.content.strip()[2:]


class MarkdownExtractor:
    def __init__(self, file_type: FileType, file_path: str):
        self.file_type = file_type
        self.file_path = file_path

        self.blocks: List[ContentBlock] = []
        self.current_code_block = None

    def ingest(self, line: str):
        stripped_line = line.strip()

        # Case of Markdown line
        if stripped_line.startswith("#"):
            self.current_code_block = None
            self.blocks.append(MarkdownBlock(line))
        else:
            if self.current_code_block is None:
                self.current_code_block = CodeBlock(self.file_type, self.file_path)
                self.blocks.append(self.current_code_block)
            self.current_code_block.content += line + "\n"

    def complete(self) -> str:
        if self.current_code_block is not None:
            self.blocks.append(self.current_code_block)

        return "\n".join([block.formatted_content() for block in self.blocks])


def _extract_mdx_content(full_file_path: str, path: str) -> str:
    file_content = fetch_file_contents(full_file_path)
    file_type = _get_file_type(path)
    extractor = MarkdownExtractor(file_type, path)
    for line in file_content.splitlines():
        extractor.ingest(line)

    return extractor.complete()


def _generate_truss_example(truss_directory):
    doc_information = yaml.safe_load(
        fetch_file_contents(f"{truss_directory}/{DOC_CONFIGURATION_FILE}")
    )

    example_destination = _get_example_destination(truss_directory)
    print("Destination: ", example_destination)

    header = f"""---
title: "{doc_information["title"]}"
description: "{doc_information["description"]}"
---
"""
    files_to_scrape = doc_information["files"]

    file_content = "\n".join(
        [
            _extract_mdx_content(Path(truss_directory) / file, file)
            for file in files_to_scrape
        ]
    )
    example_content = f"""{header}\n{file_content}"""
    path_to_example = Path(example_destination)
    path_to_example.parent.mkdir(parents=True, exist_ok=True)

    path_to_example.write_text(example_content)


def _update_toc(example_dirs):
    """
    Update the table of contents in the README.md file.
    """
    print(example_dirs)
    mint_config = json.loads(fetch_file_contents(MINT_CONFIG_PATH))
    navigation = mint_config["navigation"]

    examples_group = [item for item in navigation if item["group"] == "Examples"][0]
    # examples_group["pages"].append()


def generate_truss_examples():
    clone_repo()

    example_dirs = _fetch_example_dirs(DESTINATION_DIR)
    print(f"Directories containing {DOC_CONFIGURATION_FILE}:")
    for truss_directory in example_dirs:
        _generate_truss_example(truss_directory)

    _update_toc(example_dirs)


if __name__ == "__main__":
    generate_truss_examples()
