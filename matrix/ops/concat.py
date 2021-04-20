import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def concat(*elements: AbstractMatrix, axis=0):
    if structured(*elements):
        elements_str = ", ".join(map(str, elements[:3]))
        if len(elements) > 3:
            elements_str += "..."
        warn_upmodule(
            f"Concatenating {elements_str}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.concat(*(B.dense(el) for el in elements), axis=axis))