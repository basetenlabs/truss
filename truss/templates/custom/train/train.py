from pathlib import Path


class Train:
    def __init__(
        self,
        config,
        output_model_artifacts_dir: Path,
    ):
        self.config = config
        self.training_artifacts_dir = output_model_artifacts_dir

    def train(self):
        # Write your training code here, populating generate artifacts in
        # self.training_artifacts_dir
        pass
