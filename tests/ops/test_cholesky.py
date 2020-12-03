import lab as B
import pytest

from matrix import Dense, Diagonal, Kronecker, LowerTriangular, LowRank

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    zero1,
    dense_pd,
    diag_pd,
    const_pd,
    lr_pd,
    lr1,
    wb_pd,
    kron_pd,
)


def _check_cache(a):
    chol1 = B.cholesky(a)
    chol2 = B.cholesky(a)
    assert chol1 is chol2


def test_cholesky_square_assertion():
    with pytest.raises(AssertionError):
        B.cholesky(Dense(B.randn(3, 4)))


def test_cholesky_zero(zero1):
    assert B.cholesky(zero1) is zero1


def test_cholesky_dense(dense_pd):
    check_un_op(B.cholesky, dense_pd, asserted_type=LowerTriangular)
    _check_cache(dense_pd)


def test_cholesky_diag(diag_pd):
    check_un_op(B.cholesky, diag_pd, asserted_type=Diagonal)
    _check_cache(diag_pd)


def test_cholesky_const(const_pd):
    chol = B.dense(B.cholesky(const_pd))
    approx(B.matmul(chol, chol, tr_b=True), const_pd)
    _check_cache(const_pd)


def test_cholesky_lr(lr_pd):
    chol = B.dense(B.cholesky(lr_pd))
    approx(B.matmul(chol, chol, tr_b=True), lr_pd)
    _check_cache(lr_pd)


def test_cholesky_wb(wb_pd):
    with AssertDenseWarning("converting <woodbury> to dense"):
        check_un_op(B.cholesky, wb_pd, asserted_type=LowerTriangular)


def test_cholesky_kron(kron_pd):
    check_un_op(B.cholesky, kron_pd, asserted_type=Kronecker)
    _check_cache(kron_pd)
