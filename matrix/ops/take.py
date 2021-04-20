import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def take(a: AbstractMatrix, indices_or_mask, axis=0):
    if structured(a):
        warn_upmodule(f"Taking from {a}: converting to dense.", category=ToDenseWarning)
    return B.take(B.dense(a), indices_or_mask, axis=axis)