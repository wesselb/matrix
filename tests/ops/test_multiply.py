import lab as B

from matrix import Dense, Diagonal
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_op,

    diag1,
    diag2,
    dense1,
    dense2
)


def test_multiply_dense(dense1, dense2):
    check_op(B.subtract, dense1, dense2, asserted_type=Dense)


def test_multiply_diag_dense(diag1, dense2):
    check_op(B.subtract, diag1, dense2, asserted_type=Dense)


def test_multiply_diagonal(diag1, diag2):
    check_op(B.subtract, diag1, diag2, asserted_type=Diagonal)
