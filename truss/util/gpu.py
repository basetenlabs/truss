import subprocess as sp
from typing import Optional


def get_gpu_memory() -> Optional[int]:
    # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    try:
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[1]
        )
        memory_free_values = int(memory_free_info.split()[0])
        return memory_free_values
    except FileNotFoundError:
        return None
