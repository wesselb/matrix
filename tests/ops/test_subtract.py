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


def test_subtract_dense(dense1, dense2):
    check_bin_op(B.subtract, dense1, dense2, asserted_type=Dense)


def test_subtract_diag_dense(diag1, dense2):
    check_bin_op(B.subtract, diag1, dense2, asserted_type=Dense)


def test_subtract_diagonal(diag1, diag2):
    check_bin_op(B.subtract, diag1, diag2, asserted_type=Diagonal)


def test_subtract_const_dense(const_or_scalar1, dense2):
    check_bin_op(B.subtract, const_or_scalar1, dense2, asserted_type=Dense)
    check_bin_op(B.subtract, dense2, const_or_scalar1, asserted_type=Dense)


def test_subtract_const_diag(const_or_scalar1, diag2):
    check_bin_op(B.subtract, const_or_scalar1, diag2, asserted_type=Dense)


def test_subtract_const(const_or_scalar1, const2):
    check_bin_op(B.subtract, const_or_scalar1, const2, asserted_type=Constant)
