import jax.numpy as jnp
import lab.jax as B
import numpy as np
import pytest

from matrix import Dense, Zero, Constant, LowRank, structured

# noinspection PyUnresolvedReferences
from ..util import (
    ConditionalContext,
    AssertDenseWarning,
    approx,
    check_un_op,
    const1,
    dense1,
    diag1,
    mat1,
    zero1,
    lr1,
)


def _check_take(a, asserted_type=object):
    for axis in [0, 1, -1]:
        for indices_or_mask in [
            [0, 1],
            [0, 2],
            [True, True, False, False, True, False],
        ]:
            for wrap in [list, tuple, np.array]:
                res = B.take(a, wrap(indices_or_mask), axis=axis)
                approx(res, B.take(B.dense(a), wrap(indices_or_mask), axis=axis))
                assert isinstance(res, asserted_type)


def test_take_zero(zero1):
    with pytest.raises(ValueError):
        B.take(zero1, [[1]])
    _check_take(zero1, asserted_type=Zero)


def test_take_dense(dense1):
    _check_take(dense1, asserted_type=Dense)


def test_take_diag(diag1):
    with AssertDenseWarning("taking from <diagonal>"):
        _check_take(diag1, asserted_type=Dense)


def test_take_const(const1):
    with pytest.raises(ValueError):
        B.take(const1, [[1]])
    _check_take(const1, asserted_type=Constant)


def test_take_lr(lr1):
    with ConditionalContext(
        structured(lr1.left, lr1.right),
        AssertDenseWarning("taking from <diagonal>"),
    ):
        _check_take(lr1, asserted_type=LowRank)
