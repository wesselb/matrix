import lab as B

from matrix import Dense, Diagonal, Constant
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,

    dense1,
    dense2,
    diag1,
    diag2,
    const_or_scalar1,
    const1,
    const2
)


def test_multiply_dense(dense1, dense2):
    check_bin_op(B.multiply, dense1, dense2, asserted_type=Dense)


def test_multiply_diag_dense(diag1, dense2):
    check_bin_op(B.multiply, diag1, dense2, asserted_type=Dense)


def test_multiply_diagonal(diag1, diag2):
    check_bin_op(B.multiply, diag1, diag2, asserted_type=Diagonal)


def test_multiply_const_dense(const_or_scalar1, dense1):
    check_bin_op(B.multiply, const_or_scalar1, dense1, asserted_type=Dense)


def test_multiply_const_diag(const_or_scalar1, diag1):
    check_bin_op(B.multiply, const_or_scalar1, diag1, asserted_type=Dense)


def test_multiply_const(const_or_scalar1, const2):
    check_bin_op(B.multiply, const_or_scalar1, const2, asserted_type=Constant)
