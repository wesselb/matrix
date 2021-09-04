import lab.jax as B
import numpy as np
import pytest

from matrix import (
    Dense,
    Zero,
    Constant,
    LowerTriangular,
    UpperTriangular,
    LowRank,
    structured,
)

# noinspection PyUnresolvedReferences
from ..util import (
    ConditionalContext,
    AssertDenseWarning,
    approx,
    check_un_op,
    zero1,
    const1,
    dense1,
    diag1,
    lt1,
    ut1,
    lr1,
    wb1,
    kron1,
)


def _check_take(a, asserted_type=object, warn_taking=False):
    for wrap in [list, tuple, np.array]:
        for axis in [B.rank(a) - 1, B.rank(a) - 2, -1, -2]:
            for indices_or_mask in [
                [0, 1],
                [0, 2],
                [True, True, False, False, True, False],
            ]:
                with ConditionalContext(warn_taking, AssertDenseWarning("taking from")):
                    res = B.take(a, wrap(indices_or_mask), axis=axis)
                approx(res, B.take(B.dense(a), wrap(indices_or_mask), axis=axis))
                assert isinstance(res, asserted_type)

        # Also test taking from a batch dimension!
        if B.shape_batch(a) != ():
            for axis in [0, -3]:
                for indices_or_mask in [
                    [0, 1],
                    [0, 2],
                    [True, True, False],
                ]:
                    res = B.take(a, wrap(indices_or_mask), axis=axis)
                    approx(res, B.take(B.dense(a), wrap(indices_or_mask), axis=axis))
                    # This should always preserve the type.
                    assert isinstance(res, type(a))


def test_take_zero(zero1):
    with pytest.raises(ValueError):
        B.take(zero1, [[1]])
    _check_take(zero1, asserted_type=Zero)


def test_take_dense(dense1):
    _check_take(dense1, asserted_type=Dense)


def test_take_diag(diag1):
    _check_take(diag1, asserted_type=Dense, warn_taking=True)


def test_take_const(const1):
    with pytest.raises(ValueError):
        B.take(const1, [[1]])
    _check_take(const1, asserted_type=Constant)


def test_take_lt(lt1):
    _check_take(lt1, asserted_type=Dense)


def test_take_ut(ut1):
    _check_take(ut1, asserted_type=Dense)


def test_take_lr(lr1):
    _check_take(lr1, asserted_type=LowRank, warn_taking=structured(lr1.left, lr1.right))


def test_take_wb(wb1):
    _check_take(wb1, asserted_type=Dense, warn_taking=True)


def test_take_kron(kron1):
    _check_take(kron1, asserted_type=Dense, warn_taking=True)
