if __name__ == "__main__":
    import os
    import sys

    from huggingface_hub import hf_hub_download

    # TODO(varun): might make sense to move this check + write to a separate `prepare_cache.py` script
    file_path = os.path.expanduser("~/.cache/huggingface/hub/version.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            f.write("1")

    file_name = sys.argv[1]
    repo_name = sys.argv[2]

    if len(sys.argv) >= 4:
        revision_name = sys.argv[3]
        hf_hub_download(repo_name, file_name, revision=revision_name)
    else:
        hf_hub_download(repo_name, file_name)
