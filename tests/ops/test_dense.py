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


def test_dense_zero(zero1):
    allclose(B.dense(zero1), B.zeros(zero1.rows, zero1.cols))


def test_dense_numeric(mat1):
    allclose(B.dense(mat1), mat1)


def test_dense_dense(dense1):
    allclose(B.dense(dense1), dense1.mat)


def test_dense_diag(diag1):
    allclose(B.dense(diag1), B.diag(diag1.diag))


def test_dense_const(const1):
    allclose(B.dense(const1), const1.const * B.ones(const1.rows, const1.cols))


def test_dense_lr(lr1):
    allclose(B.dense(lr1), B.outer(lr1.left, lr1.right))


def test_dense_wb(wb1):
    allclose(B.dense(wb1),
             B.diag(wb1.diag.diag) + B.outer(wb1.lr.left, wb1.lr.right))


def test_dense_kron(kron1):
    allclose(B.dense(kron1), B.kron(B.dense(kron1.left), B.dense(kron1.right)))
