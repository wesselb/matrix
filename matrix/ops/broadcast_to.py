import lab as B
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import Dimension, Shape, broadcast, get_shape
from ..util import ToDenseWarning

__all__ = []

_dispatch = Dispatcher()


@B.dispatch
def broadcast_to(a: AbstractMatrix, *shape: B.Int):
    if B.shape(a) == shape:
        return a
    else:
        return _broadcast_to(a, *shape)


@_dispatch
def _broadcast_to(a: AbstractMatrix, *shape: B.Int):
    if structured(a):
        warn_upmodule(
            f"Broadcasting {a} to shape {Shape(shape)}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.broadcast_to(B.dense(a), *shape))


@_dispatch
def _broadcast_to(a: Zero, *shape: B.Int):
    target_shape = broadcast(
        get_shape(a), Shape(*[Dimension(d) for d in shape])
    ).as_tuple()
    return Zero(a.dtype, *target_shape)


@_dispatch
def _broadcast_to(a: Constant, *shape: B.Int):
    target_shape = broadcast(
        get_shape(a), Shape(*[Dimension(d) for d in shape])
    ).as_tuple()
    return Constant(a.const, *target_shape)
