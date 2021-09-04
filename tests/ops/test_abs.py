import lab as B

from matrix import (
    Constant,
    Dense,
    Diagonal,
    Kronecker,
    LowerTriangular,
    UpperTriangular,
    Zero,
)

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    IgnoreDenseWarning,
    approx,
    check_un_op,
    const1,
    dense1,
    diag1,
    kron1,
    lr1,
    lt1,
    ut1,
    wb1,
    zero1,
)


def test_abs_zero(zero1):
    check_un_op(B.abs, zero1, asserted_type=Zero)


def test_abs_dense(dense1):
    check_un_op(B.abs, dense1 * dense1, asserted_type=Dense)


def test_abs_diag(diag1):
    check_un_op(B.abs, diag1 * diag1, asserted_type=Diagonal)


def test_abs_const(const1):
    check_un_op(B.abs, const1 * const1, asserted_type=Constant)


def test_abs_lt(lt1):
    check_un_op(B.abs, lt1 * lt1, asserted_type=LowerTriangular)


def test_abs_ut(ut1):
    check_un_op(B.abs, ut1 * ut1, asserted_type=UpperTriangular)


def test_abs_lr(lr1):
    with IgnoreDenseWarning():
        lr_pos = lr1 * lr1
    with AssertDenseWarning("absolute value of <low-rank>"):
        check_un_op(B.abs, lr_pos, asserted_type=Dense)


def test_abs_wb(wb1):
    with IgnoreDenseWarning():
        wb_pos = wb1 * wb1
    with AssertDenseWarning("absolute value of <woodbury>"):
        check_un_op(B.abs, wb_pos, asserted_type=Dense)


def test_abs_kron(kron1):
    check_un_op(B.abs, kron1 * kron1, asserted_type=Kronecker)
