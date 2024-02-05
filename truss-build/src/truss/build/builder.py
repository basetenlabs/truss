from .image import Image
import tempfile
from pathlib import Path


def build(image: Image):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".Dockerfile"
    ) as temp_dockerfile:
        dockerfile_path = temp_dockerfile.name
        print(f"Dockerfile created at: {dockerfile_path}")
        Path(dockerfile_path).write_text(image.serialize())
