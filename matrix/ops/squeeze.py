import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def squeeze(a: AbstractMatrix):
    if structured(a):
        warn_upmodule(f"Squeezing {a}: converting to dense.", category=ToDenseWarning)
    return B.squeeze(B.dense(a))
