import lab as B

from matrix import Dense, Diagonal, Constant
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense1,
    diag1,
    const1
)


def power2(x):
    return B.power(x, 2)


def test_power_dense(dense1):
    check_un_op(power2, dense1, asserted_type=Dense)


def test_power_diag(diag1):
    check_un_op(power2, diag1, asserted_type=Diagonal)


def test_power_const(const1):
    check_un_op(power2, const1, asserted_type=Constant)
