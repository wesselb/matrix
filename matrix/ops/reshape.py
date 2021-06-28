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
    a = B.reshape(B.dense(a), *shape)
    if len(shape) == 2:
        return Dense(a)
    else:
        return a


@B.dispatch
def reshape(a: Dense, rows: B.Int, cols: B.Int):
    return Dense(B.reshape(a.mat, rows, cols))
