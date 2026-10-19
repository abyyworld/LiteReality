"""Stage protocol and declarative metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult


class StageCallable(Protocol):
    def __call__(self, context: RunContext, options: dict[str, Any]) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    run: StageCallable
    prerequisites: tuple[str, ...] = ()
    required: bool = True
    is_complete: Callable[[RunContext], bool] | None = None
    # A stage whose inputs depend on how it is asked to run. `simulate --from-seed` exports the room
    # scene init built, so demanding `publish` (the authored room) would refuse the one mode that
    # exists to skip authoring. When set, it replaces `prerequisites` for that run's options.
    prerequisites_for: Callable[[dict[str, Any]], tuple[str, ...]] | None = None

    def requires(self, options: dict[str, Any]) -> tuple[str, ...]:
        return self.prerequisites_for(options) if self.prerequisites_for else self.prerequisites
