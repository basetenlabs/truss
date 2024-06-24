import os
import urllib.request


def _download_files(urls):
    ASSETS_DIR = os.path.join(
        os.path.expanduser("~"), ".cache", "whisper-trt", "assets"
    )
    os.makedirs(ASSETS_DIR, exist_ok=True)

    for url in urls:
        file_name = os.path.basename(url)
        file_path = os.path.join(ASSETS_DIR, file_name)
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
    return ASSETS_DIR


def download_assets():
    return _download_files(
        [
            "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
            "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
        ]
    )
