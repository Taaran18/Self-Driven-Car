import inspect
import textwrap
from functools import lru_cache
from pathlib import Path

from app.simulation.engine import GenerationRun, decide, move, score, sense, think
from app.simulation.evolution import evolve, summarize_generation
from app.simulation.geometry import cast_rays
from app.simulation.trainer import Trainer

_ROOT = Path(__file__).resolve().parents[2]

STEPS = [
    ("train", [Trainer.train]),
    ("tick", [GenerationRun.tick]),
    ("sense", [sense, cast_rays]),
    ("think", [think]),
    ("decide", [decide]),
    ("move", [move]),
    ("score", [score]),
    ("evaluate", [summarize_generation]),
    ("evolve", [evolve]),
]


def _snippet(fn) -> dict:
    lines, start = inspect.getsourcelines(fn)
    file = Path(inspect.getsourcefile(fn) or "").resolve()
    return {
        "name": fn.__qualname__,
        "file": str(file.relative_to(_ROOT)) if file.is_relative_to(_ROOT) else file.name,
        "start_line": start,
        "code": textwrap.dedent("".join(lines)).rstrip(),
    }


@lru_cache
def code_steps() -> list[dict]:
    return [{"id": step_id, "snippets": [_snippet(fn) for fn in functions]} for step_id, functions in STEPS]
