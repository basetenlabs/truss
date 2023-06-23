from pathlib import Path
from typing import Dict


class Train:
    def __init__(
        self,
        config,
        output_dir: Path,
        variables: Dict,
        secrets,
    ):
        self._config = config
        self._output_dir = output_dir
        self._variables = variables
        self._secrets = secrets

    def train(self):
        # Write your training code here, populating generated artifacts in
        # self._output_dir
        pass
