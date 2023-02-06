from pathlib import Path
from typing import Dict, List

from truss.constants import CONFIG_FILE
from truss.multi_truss.config import MultiTrussConfig
from truss.truss_handle import TrussHandle


class MultiTrussSpec:
    def __init__(self, multi_truss_dir: Path) -> None:
        self.dir = multi_truss_dir
        self.config = MultiTrussConfig.from_yaml(multi_truss_dir / CONFIG_FILE)

    @property
    def trusses_names(self) -> List[str]:
        return list([t.name for t in self.config.trusses])

    @property
    def trusses_paths(self) -> List[str]:
        return list([t.path for t in self.config.trusses])

    @property
    def trusses_dir_paths(self) -> List[Path]:
        paths = []
        for path_name in self.trusses_paths:
            path = Path(path_name)
            if path.is_absolute():
                paths.append(path)
            else:
                paths.append(self.dir / path)
        return paths

    @property
    def prepared_truss_dir_paths(self) -> Dict[str, Path]:
        # Make sure that all the children trusses are ready to be copied
        return dict(
            {
                name: TrussHandle(truss_path, validate=True).gather()
                for name, truss_path in zip(self.trusses_names, self.trusses_dir_paths)
            }
        )

    def update_resources(self) -> None:
        # TODO: Update GPU and Memory to accumulate using k8s resources
        self.config.resources.cpu = 0
        self.config.resources.memory = 0
        self.config.resources.use_gpu = False
        for truss_path in self.trusses_dir_paths:
            config = TrussHandle(truss_path, validate=True).spec.config
            self.config.resources.use_gpu |= config.resources.use_gpu
