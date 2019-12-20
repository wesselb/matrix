import lab as B
import pytest

from matrix import Dense, Diagonal, Kronecker, Zero
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    zero1,
    dense_pd,
    diag_pd,
    const_pd,
    lr_pd,
    lr1,
    kron_pd
)


def test_cholesky_square_assertion():
    with pytest.raises(AssertionError):
        B.cholesky(Dense(B.randn(3, 4)))


def test_cholesky_zero(zero1):
    check_un_op(B.cholesky, zero1, asserted_type=Zero)


def test_cholesky_dense(dense_pd):
    check_un_op(B.cholesky, dense_pd, asserted_type=Dense)


def test_cholesky_diag(diag_pd):
    check_un_op(B.cholesky, diag_pd, asserted_type=Diagonal)


def test_cholesky_const(const_pd):
    chol = B.dense(B.cholesky(const_pd))
    allclose(B.matmul(chol, chol, tr_b=True), const_pd)


def test_cholesky_lr(lr_pd, lr1):
    chol = B.dense(B.cholesky(lr_pd))
    allclose(B.matmul(chol, chol, tr_b=True), lr_pd)

    with pytest.raises(AssertionError):
        B.cholesky(lr1)


def test_cholesky_kron(kron_pd):
    check_un_op(B.cholesky, kron_pd, asserted_type=Kronecker)
