import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    mat1,
    dense1,
    diag1
)


def test_dense_numeric(mat1):
    allclose(B.dense(mat1), mat1)


def test_dense_dense(dense1):
    allclose(B.dense(dense1), dense1.mat)


def test_dense_diagonal(diag1):
    allclose(B.dense(diag1), B.diag(diag1.diag))
