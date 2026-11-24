"""Task 1 rubric: dimension maxima and a validated score container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

DIMS = {"compilable": 1, "io_format": 1, "logic": 4, "edge_case": 2, "complexity": 1, "code_quality": 1}


@dataclass(frozen=True)
class Rubric:
    compilable: int
    io_format: int
    logic: int
    edge_case: int
    complexity: int
    code_quality: int

    def __post_init__(self) -> None:
        # Every dimension must be an int in 0..DIMS[name]; reject anything else instead of clamping.
        for f in fields(self):
            value, mx = getattr(self, f.name), DIMS[f.name]
            if type(value) is not int or not 0 <= value <= mx:
                raise ValueError(f"{f.name}={value!r} not an int in [0, {mx}]")

    @classmethod
    def from_dict(cls, scores: dict) -> Rubric:
        """Build from a parsed reply; extra keys are ignored, a missing dimension raises KeyError."""
        return cls(**{d: scores[d] for d in DIMS})

    @property
    def total(self) -> int:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
