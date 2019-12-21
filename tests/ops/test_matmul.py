import pytest
import lab as B

from matrix import Dense, Diagonal, Zero, Constant, LowRank, Woodbury, Kronecker
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,

    zero1,
    zero2,
    zero_r,
    dense1,
    dense2,
    diag1,
    diag2,
    const_or_scalar1,
    const_or_scalar2,
    const1,
    const2,
    lr1,
    lr2,
    wb1,
    wb2,
    kron1,
    kron2
)


def _check_matmul(a, b, asserted_type=object):
    for tr_a in [False, True]:
        for tr_b in [False, True]:
            def matmul(a, b):
                return B.matmul(a, b, tr_a=tr_a, tr_b=tr_b)

            check_bin_op(matmul, a, b, asserted_type=asserted_type)


def test_matmul_assertion(zero_r, dense2):
    with pytest.raises(AssertionError):
        B.matmul(zero_r, dense2)
    with pytest.raises(AssertionError):
        B.matmul(zero_r, dense2, tr_b=True)
    with pytest.raises(AssertionError):
        B.matmul(zero_r, zero_r, tr_a=True, tr_b=True)


def test_matmul_zero_diag(zero1, diag2):
    _check_matmul(zero1, diag2, asserted_type=Zero)
    _check_matmul(diag2, zero1, asserted_type=Zero)


def test_matmul_dense(dense1, dense2):
    _check_matmul(dense1, dense2, asserted_type=Dense)


def test_matmul_diag(diag1, diag2):
    _check_matmul(diag1, diag2, asserted_type=Diagonal)
