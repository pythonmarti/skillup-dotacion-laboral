"""Tipos base para pipelines multi-dominio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


StageRunner = Callable[[object], None]


@dataclass(frozen=True)
class DomainPipeline:
    """Describe un pipeline de dominio con stages estandarizados."""

    name: str
    description: str
    generate: StageRunner | None = None
    etl: StageRunner | None = None
    train: StageRunner | None = None
    infer: StageRunner | None = None
    report: StageRunner | None = None

    def run_stage(self, stage: str, args: object) -> None:
        stage_map = {
            "generate": self.generate,
            "etl": self.etl,
            "train": self.train,
            "infer": self.infer,
            "report": self.report,
        }
        runner = stage_map.get(stage)
        if runner is None:
            raise NotImplementedError(
                f"El dominio '{self.name}' aun no implementa el stage '{stage}'"
            )
        runner(args)

    def run_full(self, args: object) -> None:
        for stage in ["generate", "etl", "train", "infer", "report"]:
            if getattr(self, stage) is not None:
                self.run_stage(stage, args)
