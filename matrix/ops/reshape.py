import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix, B.Int, B.Int)
def reshape(a, rows, cols):
    warn_upmodule(f"Converting {a} to dense for reshaping.", category=ToDenseWarning)
    return Dense(B.reshape(B.dense(a), rows, cols))


@B.dispatch(Dense, B.Int, B.Int)
def reshape(a, rows, cols):
    return Dense(B.reshape(a.mat, rows, cols))
