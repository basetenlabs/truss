__version__ = "0.1.0"

from truss.build.image import Image

if __name__ == "__main__":
    from pprint import pprint
    from pathlib import Path

    img = (
        Image()
        .apt_install("python3.10-venv")
        .pip_install("numpy", "torch")
        .env({"NOPROXY": "*"})
    )
    print(img.serialize())
