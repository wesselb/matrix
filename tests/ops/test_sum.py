import pytest
import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury, Kronecker, Zero

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    zero_r,
    dense_r,
    diag1,
    const_r,
    lt1,
    ut1,
    lr1,
    lr_r,
    wb1,
    kron_r,
)


def _check_sum(a):
    for axis in [None, 0, 1]:

        def sum(a):
            return B.sum(a, axis=axis)

        check_un_op(sum, a)

    with pytest.raises(ValueError):
        B.sum(a, axis=2)


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
    _check_sum(lr_r)


def test_sum_wb(wb1):
    _check_sum(wb1)


def test_sum_kron(kron_r):
    _check_sum(kron_r)
