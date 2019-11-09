import lab as B

from matrix import Dense, Diagonal
# noinspection PyUnresolvedReferences
from ..util import allclose, mat, vec


def test_dense_numeric(mat):
    allclose(B.dense(mat), mat)


def test_dense_dense(mat):
    allclose(B.dense(Dense(mat)), mat)


def test_dense_diagonal(vec):
    allclose(B.dense(Diagonal(vec)), B.diag(vec))
