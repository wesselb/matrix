import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def divide(a: AbstractMatrix, b: AbstractMatrix):
    if structured(a, b):
        warn_upmodule(
            f"Dividing {a} by {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.divide(B.dense(a), B.dense(b)))