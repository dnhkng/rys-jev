"""RYS-JEV: layer-duplicated StartLux-Decision Instant models.

    from rys_jev import load
    m = load("dnhkng/RYS-JEV")
    print(m.score({"ticket": "I was charged twice."},
                  {"team": {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "Payments", "technical": "App"}}}))
"""

__version__ = "0.1.0"

from .engine import DecisionModel, forward, enable  # noqa: F401
from .materialize import materialize  # noqa: F401


def load(path: str, device: str = "cuda:0", dtype: str = "bfloat16", check: bool = True) -> DecisionModel:
    """Load a decision checkpoint and enable bidirectional prompt attention."""
    return DecisionModel(path, device=device, dtype=dtype, check=check)
