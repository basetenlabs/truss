import traceback
from pathlib import Path
from typing import List, Optional

import pydantic


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


class StackFrame(pydantic.BaseModel):
    filename: str
    lineno: int
    name: str
    line: str

    @classmethod
    def from_frame_summary(cls, frame: traceback.FrameSummary):
        return cls(
            filename=frame.filename,
            lineno=frame.lineno,
            name=frame.name,
            line=frame.line,
        )

    def to_frame_summary(self) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            filename=self.filename, lineno=self.lineno, name=self.name, line=self.line
        )


class TrussError(pydantic.BaseModel):
    exception_class_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: List[StackFrame]

    def to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        stack = "".join(traceback.format_list(self.to_stack_summary()))
        exception_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = f"""{TrussError.__name__} (server-side)
Traceback (most recent call last):
{stack}{self.exception_class_name}: {self.exception_message}{exception_info}
"""
        return error
