import lab as B

from matrix import Dense, Diagonal
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense1,
    diag1
)


def power2(x):
    return B.power(x, 2)


def test_negative_dense(dense1):
    check_un_op(power2, dense1, asserted_type=Dense)


def test_negative_diag(diag1):
    check_un_op(power2, diag1, asserted_type=Diagonal)
