import lab as B
from lab.shape import Shape
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import expand_and_broadcast
from ..util import ToDenseWarning

__all__ = []

_dispatch = Dispatcher()


@B.dispatch
def broadcast_to(a: AbstractMatrix, *shape: B.Int):
    # If the shape is already right, do nothing.
    if B.shape(a) == shape:
        return a
    else:
        return _broadcast_to(a, *shape)


@_dispatch
def _broadcast_to(a: AbstractMatrix, *shape: B.Int):
    # Check whether the matrix shapes align. If they do, defer the job to
    # `broadcast_batch_to`, which is always able to preserve the matrix type.
    if B.shape(a)[-2:] == shape[-2:]:
        return B.broadcast_batch_to(a, *shape[:-2])
    else:
        if structured(a):
            warn_upmodule(
                f"Broadcasting {a} to shape {Shape(*shape)}: converting to dense.",
                category=ToDenseWarning,
            )
        return Dense(B.broadcast_to(B.dense(a), *shape))


@_dispatch
def _broadcast_to(a: Zero, *shape: B.Int):
    return Zero(a.dtype, *expand_and_broadcast(B.shape(a), Shape(*shape)))


@_dispatch
def _broadcast_to(a: Constant, *shape: B.Int):
    return Constant(a.const, *expand_and_broadcast(B.shape(a), Shape(*shape)))
