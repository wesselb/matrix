import lab as B
import pytest

from matrix import Dense, Diagonal, Kronecker, LowerTriangular, LowRank

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const_pd,
    dense1_pd,
    diag1_pd,
    kron_pd,
    lr1,
    lr1_pd,
    wb1_pd,
    zero1,
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


def test_cholesky_dense(dense1_pd):
    check_un_op(B.cholesky, dense1_pd, asserted_type=LowerTriangular)
    _check_cache(dense1_pd)


def test_cholesky_diag(diag1_pd):
    check_un_op(B.cholesky, diag1_pd, asserted_type=Diagonal)
    _check_cache(diag1_pd)


def test_cholesky_const(const_pd):
    chol = B.dense(B.cholesky(const_pd))
    approx(B.matmul(chol, chol, tr_b=True), const_pd)
    _check_cache(const_pd)


def test_cholesky_lr(lr1_pd):
    chol = B.dense(B.cholesky(lr1_pd))
    approx(B.matmul(chol, chol, tr_b=True), lr1_pd)
    _check_cache(lr1_pd)


def test_cholesky_wb(wb1_pd):
    with AssertDenseWarning("converting <woodbury> to dense"):
        check_un_op(B.cholesky, wb1_pd, asserted_type=LowerTriangular)


def test_cholesky_kron(kron_pd):
    check_un_op(B.cholesky, kron_pd, asserted_type=Kronecker)
    _check_cache(kron_pd)
