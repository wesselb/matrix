import jax.numpy as jnp
import lab.jax as B
import numpy as np
import pytest

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const1,
    dense1,
    diag1,
    mat1,
    zero1,
)


def _check_take(a):
    for axis in [0, 1, -1]:
        for indices_or_mask in [
            [0, 1],
            [0, 2],
            [True, True, False, False, True, False],
        ]:
            for wrap in [list, tuple, np.array]:
                approx(
                    B.take(a, wrap(indices_or_mask), axis=axis),
                    B.take(B.dense(a), wrap(indices_or_mask), axis=axis),
                )


def test_take_jit(mat1):
    @B.jit
    def take_jitted(a):
        return B.take(a, [0, 1])

    approx(take_jitted(jnp.array(mat1)), B.take(B.dense(mat1), [0, 1]))


def test_take_zero(zero1):
    with pytest.raises(ValueError):
        B.take(zero1, [[1]])
    _check_take(zero1)


def test_take_dense(dense1):
    _check_take(dense1)


def test_take_diag(diag1):
    with AssertDenseWarning("taking from <diagonal>"):
        _check_take(diag1)


def test_take_const(const1):
    with pytest.raises(ValueError):
        B.take(const1, [[1]])
    _check_take(const1)
