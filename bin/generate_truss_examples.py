"""
Script to take the Truss examples in https://github.com/basetenlabs/truss-examples,
and generate documentation.

Usage:
```
$ poetry run python bin/generate_truss_examples.py
```
"""
import enum
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

DOC_CONFIGURATION_FILE = "doc.yaml"
TRUSS_EXAMPLES_REPO = "https://github.com/basetenlabs/truss-examples"
DESTINATION_DIR = "truss-examples"
MINT_CONFIG_PATH = "docs/mint.json"


class FileType(enum.Enum):
    YAML = "yaml"
    PYTHON = "python"


def clone_repo():
    """
    If the destination directory exists, remove it.
    Then, clone the given repo into the specified directory.
    """
    if Path(DESTINATION_DIR).exists():
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


def _fetch_example_dirs(root_dir: str) -> List[str]:
    """
    Walk through the directory structure from the root directory and
    find all directories that have the specified file in it.
    """
    dirs_with_file = []

    for dirpath, _, filenames in os.walk(root_dir):
        if DOC_CONFIGURATION_FILE in filenames:
            dirs_with_file.append(dirpath)

    return dirs_with_file


def _get_example_destination(truss_directory: str) -> Path:
    """
    Get the destination directory for the example.
    """
    original_path = Path(truss_directory)
    folder, example = original_path.parts[1:]
    example_file = f"{example}.mdx"
    return Path("docs/examples") / folder / example_file


def _get_file_type(file_path: str) -> FileType:
    extension = Path(file_path).suffix
    if extension == ".yaml":
        return FileType.YAML

    if extension == ".py":
        return FileType.PYTHON

    raise ValueError(f"Unknown file type: {extension}")


class ContentBlock:
    def formatted_content(self) -> str:
        raise NotImplementedError


class CodeBlock(ContentBlock):
    def __init__(self, file_type: FileType, file_path: str):
        self.file_type = file_type
        self.file_path = file_path
        self.content = ""

    def formatted_content(self) -> str:
        """
        Outputs code blocks in the format:

        ```python main.py
        def main():
            ...
        ```
        """
        return f"\n```{self.file_type.value} {self.file_path}\n{self.content}```"


class MarkdownBlock(ContentBlock):
    def __init__(self, content: str):
        self.content = content

    def formatted_content(self) -> str:
        # Remove the first comment and space character, such that
        # "# Hello" becomes "Hello
        return self.content.strip()[2:]


class MarkdownExtractor:
    """
    Class that supports ingesting a code file line-by-line, and produces a formatted
    mdx file.
    """

    def __init__(self, file_type: FileType, file_path: str):
        self.file_type = file_type
        self.file_path = file_path

        self.blocks: List[ContentBlock] = []
        self.current_code_block: Optional[CodeBlock] = None

    def ingest(self, line: str):
        """
        For each line, check that it is a comment by the presence of "#".
        If it is a comment, append it to the blocks.

        If it is not a comment, either append to the current code block, or
        create a new code block if this isn't one.

        When this is finished, we can then very easily produce the mdx file.
        """
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

    def _formatted_request_example(self) -> str:
        """
        A key part of the mdx file is that each has a <RequestExample> block at the
        bottom the file. This generates that for the given file by appending all the
        CodeBlocks together.
        """
        code_blocks = [block for block in self.blocks if isinstance(block, CodeBlock)]
        code_content = "".join([code_block.content for code_block in code_blocks])

        return f"""```{self.file_type.value} {self.file_path}\n{code_content}```"""

    def mdx_content(self) -> Tuple[str, str]:
        full_content = "\n".join([block.formatted_content() for block in self.blocks])

        return (
            full_content + "\n",
            self._formatted_request_example(),
        )


def _extract_mdx_content_and_code(full_file_path: str, path: str) -> Tuple[str, str]:
    file_content = fetch_file_contents(full_file_path)
    file_type = _get_file_type(path)
    extractor = MarkdownExtractor(file_type, path)
    for line in file_content.splitlines():
        extractor.ingest(line)

    return extractor.mdx_content()


def _generate_request_example_block(code: str):
    return f"""
<RequestExample>
{code}
</RequestExample>
"""


def _generate_truss_example(truss_directory: str):
    print("Generating example for: ", truss_directory)
    doc_information = yaml.safe_load(
        fetch_file_contents(f"{truss_directory}/{DOC_CONFIGURATION_FILE}")
    )

    example_destination = _get_example_destination(truss_directory)

    header = f"""---
title: "{doc_information["title"]}"
description: "{doc_information["description"]}"
---
"""

    path_in_examples_repo = "/".join(Path(truss_directory).parts[1:])
    link_to_github = f"""
        <Card
          title="View on Github"
          icon="github" href="{TRUSS_EXAMPLES_REPO}/tree/main/{path_in_examples_repo}">
        </Card>
    """
    files_to_scrape = doc_information["files"]

    full_content, code_blocks = zip(
        *[
            _extract_mdx_content_and_code(Path(truss_directory) / file, file)
            for file in files_to_scrape
        ]
    )

    full_code_block = "\n".join(code_blocks)
    file_content = "\n".join(full_content) + _generate_request_example_block(
        full_code_block
    )
    example_content = f"""{header}\n{link_to_github}\n{file_content}"""
    path_to_example = Path(example_destination)
    path_to_example.parent.mkdir(parents=True, exist_ok=True)

    path_to_example.write_text(example_content)


def _format_group_name(group_name: str) -> str:
    """
    This function takes the parent directory name in, and converts it
    into a more human readable format for the table of contents.

    Note that parent directory names are assumed to be in the format:
    * 1_introduction/... (becomes "Introduction")
    * 2_image_classification/... (becomes "Image classification")
    * 3_llms/... (becomes "LLMs")
    """
    lowercase_name = " ".join(group_name.split("_")[1:])
    # Capitalize the first letter. We do this rather than
    # use .capitalize() or .title() because we want to preserve
    # the case of subsequent letters
    return lowercase_name[0].upper() + lowercase_name[1:]


def update_toc(example_dirs: List[str]):
    """
    Update the table of contents in the README.md file.

    Parameters:
    example_dirs: List of directories as strings in the form "truss-examples-2/..."
    """

    # Exclude the root directory ("truss_examples") from the path
    transformed_example_paths = [Path(example).parts[1:] for example in example_dirs]

    mint_config = json.loads(fetch_file_contents(MINT_CONFIG_PATH))
    navigation = mint_config["navigation"]

    examples_section = [item for item in navigation if item["group"] == "Examples"][0]

    # Sort examples by the group name
    examples_section["pages"] = [
        f"examples/{example_path[0]}/{example_path[1]}"
        for example_path in sorted(
            transformed_example_paths, key=lambda example: example[0]
        )
    ]

    serialized_mint_config = json.dumps(mint_config, indent=2)
    Path(MINT_CONFIG_PATH).write_text(serialized_mint_config)


def generate_truss_examples():
    """
    Walk through the Truss examples repo, and for each
    of the examples in the repo, generate documentation.

    Finish the process by updating the table of contents.
    """
    clone_repo()

    example_dirs = _fetch_example_dirs(DESTINATION_DIR)
    for truss_directory in example_dirs:
        _generate_truss_example(truss_directory)

    update_toc(example_dirs)


if __name__ == "__main__":
    generate_truss_examples()
