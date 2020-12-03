import lab as B

from matrix import (
    Dense,
    Diagonal,
    Zero,
    Constant,
    LowerTriangular,
    UpperTriangular,
    Kronecker,
)

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    IgnoreDenseWarning,
    AssertDenseWarning,
    zero1,
    dense1,
    diag1,
    const1,
    lt1,
    ut1,
    lr1,
    wb1,
    kron1,
)


def test_sqrt_zero(zero1):
    check_un_op(B.sqrt, zero1, asserted_type=Zero)


def test_sqrt_dense(dense1):
    check_un_op(B.sqrt, dense1 * dense1, asserted_type=Dense)


def test_sqrt_diag(diag1):
    check_un_op(B.sqrt, diag1 * diag1, asserted_type=Diagonal)


def test_sqrt_const(const1):
    check_un_op(B.sqrt, const1 * const1, asserted_type=Constant)


def test_sqrt_lt(lt1):
    check_un_op(B.sqrt, lt1 * lt1, asserted_type=LowerTriangular)


def test_sqrt_ut(ut1):
    check_un_op(B.sqrt, ut1 * ut1, asserted_type=UpperTriangular)


def test_sqrt_lr(lr1):
    with IgnoreDenseWarning():
        lr_pos = lr1 * lr1
    with AssertDenseWarning("square root of <low-rank>"):
        check_un_op(B.sqrt, lr_pos, asserted_type=Dense)


def test_sqrt_wb(wb1):
    with IgnoreDenseWarning():
        wb_pos = wb1 * wb1
    with AssertDenseWarning("square root of <woodbury>"):
        check_un_op(B.sqrt, wb_pos, asserted_type=Dense)


def test_sqrt_kron(kron1):
    check_un_op(B.sqrt, kron1 * kron1, asserted_type=Kronecker)
