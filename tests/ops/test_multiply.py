import lab as B
import pytest

from matrix import (
    AbstractMatrix,
    Constant,
    Dense,
    Diagonal,
    Kronecker,
    LowerTriangular,
    LowRank,
    UpperTriangular,
    Woodbury,
    Zero,
    structured,
)

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    ConditionalContext,
    approx,
    check_bin_op,
    const1,
    const2,
    const_or_scalar1,
    const_or_scalar2,
    dense1,
    dense2,
    dense_bc,
    diag1,
    diag2,
    kron1,
    kron2,
    lr1,
    lr2,
    lr1_pd,
    lt1,
    lt2,
    ut1,
    ut2,
    wb1,
    wb2,
    zero1,
    zero2,
)


def _conditional_warning(mats, message):
    mats = [mat.left for mat in mats] + [mat.right for mat in mats]
    return ConditionalContext(structured(*mats), AssertDenseWarning(message))


def test_multiply_zero_dense(zero1, dense_bc):
    check_bin_op(B.multiply, zero1, dense_bc, asserted_type=Zero)
    check_bin_op(B.multiply, dense_bc, zero1, asserted_type=Zero)


def test_multiply_zero_diag(zero1, diag2):
    check_bin_op(B.multiply, zero1, diag2, asserted_type=Zero)
    check_bin_op(B.multiply, diag2, zero1, asserted_type=Zero)


def test_multiply_dense(dense1, dense2):
    check_bin_op(B.multiply, dense1, dense2, asserted_type=Dense)


def test_multiply_diag(diag1, diag2):
    check_bin_op(B.multiply, diag1, diag2, asserted_type=Diagonal)


def test_multiply_diag_dense(diag1, dense_bc):
    check_bin_op(B.multiply, diag1, dense_bc, asserted_type=Diagonal)
    check_bin_op(B.multiply, dense_bc, diag1, asserted_type=Diagonal)


def test_multiply_const(const_or_scalar1, const2):
    check_bin_op(B.multiply, const_or_scalar1, const2, asserted_type=Constant)


def test_multiply_const_dense(const_or_scalar1, dense_bc):
    check_bin_op(B.multiply, const_or_scalar1, dense_bc, asserted_type=Dense)
    check_bin_op(B.multiply, dense_bc, const_or_scalar1, asserted_type=Dense)


def test_multiply_const_fallback_warning(const1, diag2):
    with AssertDenseWarning("multiplying <constant> and <diagonal>"):
        B.multiply.invoke(Constant, AbstractMatrix)(const1, diag2)


def test_multiply_const_diag(const_or_scalar1, diag2):
    check_bin_op(B.multiply, const_or_scalar1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, const_or_scalar1, asserted_type=Diagonal)


def test_multiply_const_broadcasting():
    assert B.shape(B.multiply(Constant(1, 3, 4), Constant(1, 1, 4))) == (3, 4)
    assert B.shape(B.multiply(Constant(1, 3, 4), Constant(1, 3, 1))) == (3, 4)
    with pytest.raises(AssertionError):
        B.multiply(Constant(1, 3, 4), Constant(1, 4, 4))
        B.multiply(Constant(1, 3, 4), Constant(1, 3, 3))


def test_multiply_lt(lt1, lt2):
    check_bin_op(B.multiply, lt1, lt2, asserted_type=LowerTriangular)


def test_multiply_lt_dense(lt1, dense2):
    check_bin_op(B.multiply, lt1, dense2, asserted_type=LowerTriangular)
    check_bin_op(B.multiply, dense2, lt1, asserted_type=LowerTriangular)


def test_multiply_lt_diag(lt1, diag2):
    check_bin_op(B.multiply, lt1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, lt1, asserted_type=Diagonal)


def test_multiply_lt_const(lt1, const_or_scalar2):
    check_bin_op(B.multiply, lt1, const_or_scalar2, asserted_type=LowerTriangular)
    check_bin_op(B.multiply, const_or_scalar2, lt1, asserted_type=LowerTriangular)


def test_multiply_ut(ut1, ut2):
    check_bin_op(B.multiply, ut1, ut2, asserted_type=UpperTriangular)


def test_multiply_ut_lt(ut1, lt2):
    check_bin_op(B.multiply, ut1, lt2, asserted_type=Diagonal)
    check_bin_op(B.multiply, lt2, ut1, asserted_type=Diagonal)


def test_multiply_ut_dense(ut1, dense2):
    check_bin_op(B.multiply, ut1, dense2, asserted_type=UpperTriangular)
    check_bin_op(B.multiply, dense2, ut1, asserted_type=UpperTriangular)


def test_multiply_ut_diag(ut1, diag2):
    check_bin_op(B.multiply, ut1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, ut1, asserted_type=Diagonal)


def test_multiply_ut_const(ut1, const_or_scalar2):
    check_bin_op(B.multiply, ut1, const_or_scalar2, asserted_type=UpperTriangular)
    check_bin_op(B.multiply, const_or_scalar2, ut1, asserted_type=UpperTriangular)


lr_warnings = [
    "getting the diagonal of <low-rank>",
    "multiplying <low-rank> and <low-rank>",
]


def test_multiply_lr(lr1, lr2):
    with _conditional_warning([lr1, lr2], lr_warnings):
        check_bin_op(B.multiply, lr1, lr2, asserted_type=LowRank)


def test_multiply_lr_const(lr1, const_or_scalar2):
    check_bin_op(B.multiply, lr1, const_or_scalar2, asserted_type=LowRank)
    check_bin_op(B.multiply, const_or_scalar2, lr1, asserted_type=LowRank)


def test_multiply_lr_diag(lr1, diag2):
    with _conditional_warning([lr1], "getting the diagonal of <low-rank>"):
        check_bin_op(B.multiply, lr1, diag2, asserted_type=Diagonal)
    with _conditional_warning([lr1], "getting the diagonal of <low-rank>"):
        check_bin_op(B.multiply, diag2, lr1, asserted_type=Diagonal)


def test_multiply_lr_lt(lr1, lt2):
    check_bin_op(B.multiply, lr1, lt2, asserted_type=LowerTriangular)
    check_bin_op(B.multiply, lt2, lr1, asserted_type=LowerTriangular)


def test_multiply_lr_ut(lr1, ut2):
    check_bin_op(B.multiply, lr1, ut2, asserted_type=UpperTriangular)
    check_bin_op(B.multiply, ut2, lr1, asserted_type=UpperTriangular)


def test_multiply_wb(wb1, wb2):
    with _conditional_warning([wb1.lr, wb2.lr], lr_warnings):
        check_bin_op(B.multiply, wb1, wb2, asserted_type=Woodbury)


def test_multiply_wb_diag(wb1, diag1):
    with _conditional_warning([wb1.lr], "getting the diagonal of <low-rank>"):
        check_bin_op(B.multiply, wb1, diag1, asserted_type=Diagonal)
    with _conditional_warning([wb1.lr], "getting the diagonal of <low-rank>"):
        check_bin_op(B.multiply, diag1, wb1, asserted_type=Diagonal)


def test_multiply_wb_const(wb1, const_or_scalar2):
    check_bin_op(B.multiply, wb1, const_or_scalar2, asserted_type=Woodbury)
    check_bin_op(B.multiply, const_or_scalar2, wb1, asserted_type=Woodbury)


def test_multiply_wb_lt(wb1, lt2):
    check_bin_op(B.multiply, wb1, lt2, asserted_type=LowerTriangular)
    check_bin_op(B.multiply, lt2, wb1, asserted_type=LowerTriangular)


def test_multiply_wb_ut(wb1, ut2):
    check_bin_op(B.multiply, wb1, ut2, asserted_type=UpperTriangular)
    check_bin_op(B.multiply, ut2, wb1, asserted_type=UpperTriangular)


def test_multiply_wb_lr(wb1, lr2):
    with _conditional_warning([wb1.lr, lr2], lr_warnings):
        check_bin_op(B.multiply, wb1, lr2, asserted_type=Woodbury)
    with _conditional_warning([wb1.lr, lr2], lr_warnings):
        check_bin_op(B.multiply, lr2, wb1, asserted_type=Woodbury)


def test_multiply_kron(kron1, kron2):
    lefts_align = B.shape(kron1.left, -2, -1) == B.shape(kron2.left, -2, -1)
    rights_align = B.shape(kron1.right, -2, -1) == B.shape(kron2.right, -2, -1)
    if lefts_align and rights_align:
        check_bin_op(B.multiply, kron1, kron2, asserted_type=Kronecker)
    else:
        with pytest.raises(AssertionError):
            B.multiply(kron1, kron2)


def test_multiply_kron_dense(kron1, dense2):
    with AssertDenseWarning("multiplying <kronecker> and <dense>"):
        check_bin_op(B.multiply, kron1, dense2, asserted_type=Dense)
    with AssertDenseWarning("multiplying <dense> and <kronecker>"):
        check_bin_op(B.multiply, dense2, kron1, asserted_type=Dense)


def test_multiply_kron_diag(kron1, diag2):
    check_bin_op(B.multiply, kron1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, kron1, asserted_type=Diagonal)


def test_multiply_kron_const(kron1, const_or_scalar2):
    check_bin_op(B.multiply, kron1, const_or_scalar2, asserted_type=Kronecker)
    check_bin_op(B.multiply, const_or_scalar2, kron1, asserted_type=Kronecker)


def test_multiply_kron_lt(kron1, lt2):
    check_bin_op(B.multiply, kron1, lt2, asserted_type=LowerTriangular)
    check_bin_op(B.multiply, lt2, kron1, asserted_type=LowerTriangular)


def test_multiply_kron_ut(kron1, ut2):
    check_bin_op(B.multiply, kron1, ut2, asserted_type=UpperTriangular)
    check_bin_op(B.multiply, ut2, kron1, asserted_type=UpperTriangular)


def test_multiply_kron_lr(kron1, lr2):
    with AssertDenseWarning("multiplying <kronecker> and <low-rank>"):
        check_bin_op(B.multiply, kron1, lr2, asserted_type=Dense)
    with AssertDenseWarning("multiplying <low-rank> and <kronecker>"):
        check_bin_op(B.multiply, lr2, kron1, asserted_type=Dense)


def test_multiply_kron_wb(kron1, wb2):
    with AssertDenseWarning("multiplying <low-rank> and <kronecker>"):
        check_bin_op(B.multiply, kron1, wb2, asserted_type=Dense)
    with AssertDenseWarning("multiplying <low-rank> and <kronecker>"):
        check_bin_op(B.multiply, wb2, kron1, asserted_type=Dense)
