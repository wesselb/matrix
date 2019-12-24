import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,

    mat1,
    zero1,
    dense1,
    diag1,
    const1,
    lr1,
    wb1,
    kron1
)


def _check_cache(a):
    assert B.dense(a) is B.dense(a)


def test_dense_zero(zero1):
    allclose(B.dense(zero1), B.zeros(zero1.rows, zero1.cols))
    _check_cache(zero1)


def test_dense_numeric(mat1):
    allclose(B.dense(mat1), mat1)
    _check_cache(mat1)


def test_dense_dense(dense1):
    allclose(B.dense(dense1), dense1.mat)
    _check_cache(dense1)


def test_dense_diag(diag1):
    allclose(B.dense(diag1), B.diag(diag1.diag))
    _check_cache(diag1)


def test_dense_const(const1):
    allclose(B.dense(const1), const1.const * B.ones(const1.rows, const1.cols))
    _check_cache(const1)


def test_dense_lr(lr1):
    allclose(B.dense(lr1), B.outer(B.dense(lr1.left), B.dense(lr1.right)))
    _check_cache(lr1)


def test_dense_wb(wb1):
    allclose(B.dense(wb1),
             B.diag(wb1.diag.diag) + B.outer(B.dense(wb1.lr.left),
                                             B.dense(wb1.lr.right)))
    _check_cache(wb1)


def test_dense_kron(kron1):
    allclose(B.dense(kron1), B.kron(B.dense(kron1.left), B.dense(kron1.right)))
    _check_cache(kron1)
