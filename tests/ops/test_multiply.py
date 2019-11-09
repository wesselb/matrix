import lab as B

from matrix import Dense, Diagonal
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,

    dense1,
    dense2,
    diag1,
    diag2
)


def test_multiply_dense(dense1, dense2):
    check_bin_op(B.multiply, dense1, dense2, asserted_type=Dense)


def test_multiply_diag_dense(diag1, dense2):
    check_bin_op(B.multiply, diag1, dense2, asserted_type=Dense)


def test_multiply_diagonal(diag1, diag2):
    check_bin_op(B.multiply, diag1, diag2, asserted_type=Diagonal)
