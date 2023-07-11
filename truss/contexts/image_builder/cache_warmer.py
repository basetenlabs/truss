if __name__ == "__main__":
    import sys

    from huggingface_hub import hf_hub_download

    file_name = sys.argv[1]
    repo_name = sys.argv[2]

    if len(sys.argv) >= 4:
        revision_name = sys.argv[3]
        hf_hub_download(repo_name, file_name, revision=revision_name)
    else:
        hf_hub_download(repo_name, file_name)
