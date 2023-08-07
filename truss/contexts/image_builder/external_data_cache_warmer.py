import shutil
import sys

import requests


def download_file(file_path, url):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(file_path, "wb") as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)

    print(f"File successfully downloaded at {file_path}")


if __name__ == "__main__":
    file_path = sys.argv[1]
    url = sys.argv[2]

    download_file(file_path, url)
