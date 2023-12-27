from pathlib import Path


def create_truss(truss_dir: Path, config_contents: str, model_contents: str):
    truss_dir.mkdir(exist_ok=True)  # Ensure the 'truss' directory exists
    truss_model_dir = truss_dir / "model"
    truss_model_dir.mkdir(parents=True, exist_ok=True)

    config_file = truss_dir / "config.yaml"
    model_file = truss_model_dir / "model.py"
    with open(config_file, "w", encoding="utf-8") as file:
        file.write(config_contents)
    with open(model_file, "w", encoding="utf-8") as file:
        file.write(model_contents)
