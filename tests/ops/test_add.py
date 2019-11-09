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


def test_add_dense(dense1, dense2):
    check_op(B.add, dense1, dense2, asserted_type=Dense)


def test_add_diag_dense(diag1, dense2):
    check_op(B.add, diag1, dense2, asserted_type=Dense)


def test_add_diagonal(diag1, diag2):
    check_op(B.add, diag1, diag2, asserted_type=Diagonal)
