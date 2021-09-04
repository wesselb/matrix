import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def reshape(a: AbstractMatrix, *shape: B.Int):
    if structured(a):
        warn_upmodule(
            f"Converting {a} to dense for reshaping.", category=ToDenseWarning
        )
    return B.reshape(B.dense(a), *shape)
