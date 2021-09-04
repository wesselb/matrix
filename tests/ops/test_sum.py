import lab as B
import pytest

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    ConditionalContext,
    AssertDenseWarning,
    check_un_op,
    const_r,
    dense_r,
    diag1,
    kron_r,
    lr1,
    lr_r,
    lt1,
    ut1,
    wb1,
    zero_r,
)


def _check_sum(a, warn_batch=False):
    for axis in [None, B.rank(a) - 1, B.rank(a) - 2, -1, -2]:
        check_un_op(lambda x: B.sum(x, axis=axis), a)

    # Also test summing over the batch dimension!
    if B.shape_batch(a) != ():
        for axis in [0, -3]:
            with ConditionalContext(
                warn_batch, AssertDenseWarning("over batch dimensions")
            ):
                check_un_op(lambda x: B.sum(x, axis=axis), a)

    with pytest.raises(ValueError):
        B.sum(a, axis=5)


def test_sum_zero(zero_r):
    _check_sum(zero_r)


def test_sum_dense(dense_r):
    _check_sum(dense_r)


def test_sum_diag(diag1):
    _check_sum(diag1)


def test_sum_const(const_r):
    _check_sum(const_r)


def test_sum_lt(lt1):
    _check_sum(lt1)


def test_sum_ut(ut1):
    _check_sum(ut1)


def test_sum_lr(lr_r):
    _check_sum(lr_r, warn_batch=True)


def test_sum_wb(wb1):
    _check_sum(wb1, warn_batch=True)


def test_sum_kron(kron_r):
    _check_sum(kron_r, warn_batch=True)
