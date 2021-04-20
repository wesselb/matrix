import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def reshape(a: AbstractMatrix, rows: B.Int, cols: B.Int):
    warn_upmodule(f"Converting {a} to dense for reshaping.", category=ToDenseWarning)
    return Dense(B.reshape(B.dense(a), rows, cols))


@B.dispatch
def reshape(a: Dense, rows: B.Int, cols: B.Int):
    return Dense(B.reshape(a.mat, rows, cols))