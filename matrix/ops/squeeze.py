import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix)
def squeeze(a):
    if structured(a):
        warn_upmodule(f"Squeezing {a}: converting to dense.", category=ToDenseWarning)
    return B.squeeze(B.dense(a))
