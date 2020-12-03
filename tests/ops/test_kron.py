import lab as B

from matrix import Zero, Constant, Kronecker

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_bin_op,
    AssertDenseWarning,
    zero1,
    zero2,
    dense1,
    dense2,
    dense_bc,
    diag1,
    diag2,
    const_or_scalar1,
    const_or_scalar2,
    const1,
    const2,
    lr1,
    lr2,
    wb1,
    wb2,
    kron1,
    kron2,
)


def test_kron_zero_diag(zero1, diag2):
    check_bin_op(B.kron, zero1, diag2, asserted_type=Zero)
    check_bin_op(B.kron, diag2, zero1, asserted_type=Zero)


def test_kron_dense(dense1, dense2):
    check_bin_op(B.kron, dense1, dense2, asserted_type=Kronecker)


def test_kron_const(const1, const2):
    check_bin_op(B.kron, const1, const2, asserted_type=Constant)
