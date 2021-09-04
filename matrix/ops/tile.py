import lab as B
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


_dispatch = Dispatcher()


@B.dispatch
def tile(a: AbstractMatrix, *repetitions: B.Int):
    if structured(a):
        warn_upmodule(f"Tiling {a}: converting to dense.", category=ToDenseWarning)
    return Dense(B.tile(B.dense(a), *repetitions))


@B.dispatch
def tile(a: Zero, *repetitions: B.Int):
    if B.rank(a) != len(repetitions):
        raise ValueError(f"Must give one repetition for every dimension of {a}.")
    return Zero(
        a.dtype,
        *(a.batch[i] * repetitions[i] for i in range(len(a.batch))),
        a.rows * repetitions[-2],
        a.cols * repetitions[-1],
    )


@B.dispatch
def tile(a: Constant, *repetitions: B.Int):
    if B.rank(a) != len(repetitions):
        raise ValueError(f"Must give one repetition for every dimension of {a}.")
    return Constant(
        B.tile(a.const, *repetitions[: len(a.batch)]),
        a.rows * repetitions[-2],
        a.cols * repetitions[-1],
    )
