import lab as B
import pytest

from matrix import (
    Dense,
    Diagonal,
    Zero,
    Constant,
    LowerTriangular,
    UpperTriangular,
    LowRank,
    Woodbury,
    Kronecker
)
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,
    AssertDenseWarning,

    zero1,
    zero2,
    zero_r,
    dense1,
    dense2,
    diag1,
    diag2,
    const_or_scalar1,
    const_or_scalar2,
    const1,
    const2,
    lt1,
    lt2,
    ut1,
    ut2,
    lr1,
    lr2,
    wb1,
    wb2,
    kron1,
    kron2,
    kron_mixed
)


def _check_matmul(a, b, asserted_type=object):
    for tr_a in [False, True]:
        for tr_b in [False, True]:
            check_bin_op(lambda a_, b_: B.matmul(a_, b_, tr_a=tr_a, tr_b=tr_b),
                         a, b, asserted_type=asserted_type)


def test_matmul_assertion(zero_r, dense2):
    with pytest.raises(AssertionError):
        B.matmul(zero_r, dense2)
    with pytest.raises(AssertionError):
        B.matmul(zero_r, dense2, tr_b=True)
    with pytest.raises(AssertionError):
        B.matmul(zero_r, zero_r, tr_a=True, tr_b=True)


def test_matmul_zero_diag(zero1, diag2):
    _check_matmul(zero1, diag2, asserted_type=Zero)
    _check_matmul(diag2, zero1, asserted_type=Zero)


def test_matmul_dense(dense1, dense2):
    _check_matmul(dense1, dense2, asserted_type=Dense)


def test_matmul_diag(diag1, diag2):
    _check_matmul(diag1, diag2, asserted_type=Diagonal)


def test_matmul_diag_dense(diag1, dense2):
    _check_matmul(diag1, dense2, asserted_type=Dense)
    _check_matmul(dense2, diag1, asserted_type=Dense)


def test_matmul_const(const1, const2):
    _check_matmul(const1, const2, asserted_type=Constant)


def test_matmul_const_dense(const1, dense2):
    _check_matmul(const1, dense2, asserted_type=LowRank)
    _check_matmul(dense2, const1, asserted_type=LowRank)


def test_matmul_const_diag(const1, diag2):
    _check_matmul(const1, diag2, asserted_type=LowRank)
    _check_matmul(diag2, const1, asserted_type=LowRank)


triangular_warnings = \
    ['matrix-multiplying '
     '<lower-triangular matrix> and <upper-triangular matrix>',
     'matrix-multiplying '
     '<upper-triangular matrix> and <lower-triangular matrix>']
triangular_res_types = (LowerTriangular, UpperTriangular, Dense)


def test_matmul_lt(lt1, lt2):
    check_bin_op(B.matmul, lt1, lt2, asserted_type=LowerTriangular)
    with AssertDenseWarning(triangular_warnings):
        _check_matmul(lt1, lt2, asserted_type=triangular_res_types)


def test_matmul_lt_dense(lt1, dense2):
    _check_matmul(lt1, dense2, asserted_type=Dense)


def test_matmul_lt_diag(lt1, diag2):
    check_bin_op(B.matmul, lt1, diag2, asserted_type=LowerTriangular)
    _check_matmul(lt1, diag2, asserted_type=(LowerTriangular, UpperTriangular))
    _check_matmul(diag2, lt1, asserted_type=(LowerTriangular, UpperTriangular))


def test_matmul_lt_const(lt1, const2):
    _check_matmul(lt1, const2, asserted_type=LowRank)
    _check_matmul(const2, lt1, asserted_type=LowRank)


def test_matmul_ut(ut1, ut2):
    check_bin_op(B.matmul, ut1, ut2, asserted_type=UpperTriangular)
    with AssertDenseWarning(triangular_warnings):
        _check_matmul(ut1, ut2, asserted_type=triangular_res_types)


def test_matmul_ut_dense(ut1, dense2):
    _check_matmul(ut1, dense2, asserted_type=Dense)


def test_matmul_ut_diag(ut1, diag2):
    check_bin_op(B.matmul, ut1, diag2, asserted_type=UpperTriangular)
    _check_matmul(ut1, diag2, asserted_type=(LowerTriangular, UpperTriangular))
    _check_matmul(diag2, ut1, asserted_type=(LowerTriangular, UpperTriangular))


def test_matmul_ut_const(ut1, const2):
    _check_matmul(ut1, const2, asserted_type=LowRank)
    _check_matmul(const2, ut1, asserted_type=LowRank)


def test_matmul_ut_lt(ut1, lt1):
    res_type = (LowerTriangular, UpperTriangular, Dense)
    with AssertDenseWarning(triangular_warnings):
        _check_matmul(ut1, lt1, asserted_type=triangular_res_types)
    with AssertDenseWarning(triangular_warnings):
        _check_matmul(lt1, ut1, asserted_type=triangular_res_types)


def test_matmul_lr(lr1, lr2):
    _check_matmul(lr1, lr2, asserted_type=LowRank)
    assert B.matmul(lr1, lr2).rank == min(lr1.rank, lr2.rank)


def test_matmul_lr_dense(lr1, dense2):
    _check_matmul(lr1, dense2, asserted_type=LowRank)
    _check_matmul(dense2, lr1, asserted_type=LowRank)


def test_matmul_lr_diag(lr1, diag2):
    _check_matmul(lr1, diag2, asserted_type=LowRank)
    _check_matmul(diag2, lr1, asserted_type=LowRank)


def test_matmul_lr_const(lr1, const2):
    _check_matmul(lr1, const2, asserted_type=LowRank)
    _check_matmul(const2, lr1, asserted_type=LowRank)


def test_matmul_lr_lt(lr1, lt2):
    _check_matmul(lr1, lt2, asserted_type=LowRank)
    _check_matmul(lt2, lr1, asserted_type=LowRank)


def test_matmul_lr_ut(lr1, ut2):
    _check_matmul(lr1, ut2, asserted_type=LowRank)
    _check_matmul(ut2, lr1, asserted_type=LowRank)


def test_matmul_wb(wb1, wb2):
    _check_matmul(wb1, wb2, asserted_type=Woodbury)


def test_matmul_wb_dense(wb1, dense2):
    _check_matmul(wb1, dense2, asserted_type=Dense)
    _check_matmul(dense2, wb1, asserted_type=Dense)


def test_matmul_wb_diag(wb1, diag2):
    _check_matmul(wb1, diag2, asserted_type=Woodbury)
    _check_matmul(diag2, wb1, asserted_type=Woodbury)


def test_matmul_wb_const(wb1, const2):
    _check_matmul(wb1, const2, asserted_type=LowRank)
    _check_matmul(const2, wb1, asserted_type=LowRank)


def test_matmul_wb_lr(wb1, lr2):
    _check_matmul(wb1, lr2, asserted_type=LowRank)
    _check_matmul(lr2, wb1, asserted_type=LowRank)


wb_triangular_warnings = \
    ['adding <upper-triangular matrix> and <low-rank matrix>',
     'adding <low-rank matrix> and <upper-triangular matrix>',
     'adding <lower-triangular matrix> and <low-rank matrix>',
     'adding <low-rank matrix> and <lower-triangular matrix>']


def test_matmul_wb_lt(wb1, lt2):
    with AssertDenseWarning(wb_triangular_warnings):
        _check_matmul(wb1, lt2, asserted_type=Dense)
    with AssertDenseWarning(wb_triangular_warnings):
        _check_matmul(lt2, wb1, asserted_type=Dense)


def test_matmul_wb_ut(wb1, ut2):
    with AssertDenseWarning(wb_triangular_warnings):
        _check_matmul(wb1, ut2, asserted_type=Dense)
    with AssertDenseWarning(wb_triangular_warnings):
        _check_matmul(ut2, wb1, asserted_type=Dense)


def test_matmul_kron(kron1, kron2):
    if (
            B.shape(kron1.left)[1] == B.shape(kron2.left)[0] and
            B.shape(kron1.right)[1] == B.shape(kron2.right)[0]
    ):
        _check_matmul(kron1, kron2, asserted_type=Kronecker)
    else:
        with pytest.raises(AssertionError):
            B.matmul(kron1, kron2)


def test_matmul_kron_dense(kron_mixed, dense2):
    _check_matmul(kron_mixed, dense2, asserted_type=Dense)
    _check_matmul(dense2, kron_mixed, asserted_type=Dense)


def test_matmul_kron_const(kron1, const2):
    _check_matmul(kron1, const2, asserted_type=LowRank)
    _check_matmul(const2, kron1, asserted_type=LowRank)


def test_matmul_kron_lr(kron1, const2):
    _check_matmul(kron1, const2, asserted_type=LowRank)
    _check_matmul(const2, kron1, asserted_type=LowRank)


def test_matmul_kron_diag(kron1, diag2):
    # The output type here is dense because the product of Kronecker products
    # and diagonal matrices is dense.
    with AssertDenseWarning('cannot efficiently matrix-multiply <Kronecker '
                            'product> by <diagonal matrix>'):
        _check_matmul(kron1, diag2, asserted_type=Dense)
    with AssertDenseWarning('cannot efficiently matrix-multiply <diagonal '
                            'matrix> by <Kronecker product>'):
        _check_matmul(diag2, kron1, asserted_type=Dense)


kron_triangular_warnings = \
    ['matrix-multiplying <upper-triangular matrix> and <kronecker product>',
     'matrix-multiplying <kronecker product> and <upper-triangular matrix>',
     'matrix-multiplying <lower-triangular matrix> and <kronecker product>',
     'matrix-multiplying <kronecker product> and <lower-triangular matrix>']


def test_matmul_kron_lt(kron1, lt1):
    with AssertDenseWarning(kron_triangular_warnings):
        _check_matmul(kron1, lt1, asserted_type=Dense)
    with AssertDenseWarning(kron_triangular_warnings):
        _check_matmul(lt1, kron1, asserted_type=Dense)


def test_matmul_kron_ut(kron1, ut1):
    with AssertDenseWarning(kron_triangular_warnings):
        _check_matmul(kron1, ut1, asserted_type=Dense)
    with AssertDenseWarning(kron_triangular_warnings):
        _check_matmul(ut1, kron1, asserted_type=Dense)
