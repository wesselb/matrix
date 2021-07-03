import lab as B
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


_dispatch = Dispatcher()


@B.dispatch
def tile(a: AbstractMatrix, rows: B.Int, cols: B.Int):
    if structured(a):
        warn_upmodule(f"Tiling {a}: converting to dense.", category=ToDenseWarning)
    return Dense(B.tile(B.dense(a), rows, cols))


@B.dispatch
def tile(a: Zero, rows: B.Int, cols: B.Int):
    return Zero(a.dtype, a.rows * rows, a.cols * cols)


@B.dispatch
def tile(a: Constant, rows: B.Int, cols: B.Int):
    return Constant(a.const, a.rows * rows, a.cols * cols)
