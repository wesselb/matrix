import lab.jax as B
import pytest

from matrix import Constant, Dense, Zero

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, check_un_op, const1, dense1, diag1, zero1


def test_tile_zero(zero1):
    # Must give one repetition for every dimension.
    with pytest.raises(ValueError):
        B.tile(zero1, 1)

    repetitions = (2,) * B.rank(zero1)
    check_un_op(lambda x: B.tile(x, *repetitions), zero1, asserted_type=Zero)


def test_tile_dense(dense1):
    repetitions = (2,) * B.rank(dense1)
    check_un_op(lambda x: B.tile(x, *repetitions), dense1, asserted_type=Dense)


def test_tile_diag(diag1):
    with AssertDenseWarning("tiling <diagonal>"):
        repetitions = (2,) * B.rank(diag1)
        check_un_op(lambda x: B.tile(x, *repetitions), diag1, asserted_type=Dense)


def test_tile_const(const1):
    # Must give one repetition for every dimension.
    with pytest.raises(ValueError):
        B.tile(const1, 1)

    repetitions = (2,) * B.rank(const1)
    check_un_op(lambda x: B.tile(x, *repetitions), const1, asserted_type=Constant)
