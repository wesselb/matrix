import pytest
import lab as B
from plum import Dispatcher

from matrix import (
    Dense,
    Diagonal,
    Constant,
    LowerTriangular,
    UpperTriangular,
    LowRank,
    Woodbury,
)
from matrix import structured

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_bin_op,
    AssertDenseWarning,
    ConditionalContext,
    concat_warnings,
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
    lt1,
    lt2,
    ut1,
    ut2,
    lr1,
    lr2,
    lr_pd,
    wb1,
    wb2,
    kron1,
    kron2,
)

_dispatch = Dispatcher()


def test_add_zero_diag(zero1, diag2):
    check_bin_op(B.add, zero1, diag2, asserted_type=Diagonal)
    check_bin_op(B.add, diag2, zero1, asserted_type=Diagonal)


def test_add_dense(dense1, dense2):
    check_bin_op(B.add, dense1, dense2, asserted_type=Dense)


def test_add_diag(diag1, diag2):
    check_bin_op(B.add, diag1, diag2, asserted_type=Diagonal)


def test_add_diag_dense(diag1, dense_bc):
    check_bin_op(B.add, diag1, dense_bc, asserted_type=Dense)


def test_add_const(const_or_scalar1, const2):
    check_bin_op(B.add, const_or_scalar1, const2, asserted_type=Constant)


def test_add_const_dense(const_or_scalar1, dense2):
    # We don't test for broadcasting behaviour here because the resulting
    # shapes will be different: the constant matrix will not affect the
    # resulting shape.
    check_bin_op(B.add, const_or_scalar1, dense2, asserted_type=Dense)
    check_bin_op(B.add, dense2, const_or_scalar1, asserted_type=Dense)


def test_add_const_diag(const_or_scalar1, diag2):
    check_bin_op(B.add, const_or_scalar1, diag2, asserted_type=Woodbury)
    check_bin_op(B.add, diag2, const_or_scalar1, asserted_type=Woodbury)


def test_add_const_broadcasting():
    assert B.shape(B.add(Constant(1, 3, 4), Constant(1, 1, 4))) == (3, 4)
    assert B.shape(B.add(Constant(1, 3, 4), Constant(1, 3, 1))) == (3, 4)
    with pytest.raises(AssertionError):
        B.add(Constant(1, 3, 4), Constant(1, 4, 4))
        B.add(Constant(1, 3, 4), Constant(1, 3, 3))


def test_add_lt(lt1, lt2):
    check_bin_op(B.add, lt1, lt2, asserted_type=LowerTriangular)


def test_add_lt_dense(lt1, dense2):
    check_bin_op(B.add, lt1, dense2, asserted_type=Dense)


def test_add_lt_diag(lt1, diag2):
    check_bin_op(B.add, lt1, diag2, asserted_type=LowerTriangular)
    check_bin_op(B.add, diag2, lt1, asserted_type=LowerTriangular)


def test_add_lt_const(lt1, const_or_scalar2):
    with AssertDenseWarning("adding <constant> and <lower-triangular>"):
        check_bin_op(B.add, lt1, const_or_scalar2, asserted_type=Dense)
    with AssertDenseWarning("adding <constant> and <lower-triangular>"):
        check_bin_op(B.add, const_or_scalar2, lt1, asserted_type=Dense)


def test_add_ut(ut1, ut2):
    check_bin_op(B.add, ut1, ut2, asserted_type=UpperTriangular)


def test_add_ut_lt(ut1, lt2):
    check_bin_op(B.add, ut1, lt2, asserted_type=Dense)
    check_bin_op(B.add, lt2, ut1, asserted_type=Dense)


def test_add_ut_dense(ut1, dense2):
    check_bin_op(B.add, ut1, dense2, asserted_type=Dense)


def test_add_ut_diag(ut1, diag2):
    check_bin_op(B.add, ut1, diag2, asserted_type=UpperTriangular)
    check_bin_op(B.add, diag2, ut1, asserted_type=UpperTriangular)


def test_add_ut_const(ut1, const_or_scalar2):
    with AssertDenseWarning("adding <constant> and <upper-triangular>"):
        check_bin_op(B.add, ut1, const_or_scalar2, asserted_type=Dense)
    with AssertDenseWarning("adding <constant> and <upper-triangular>"):
        check_bin_op(B.add, const_or_scalar2, ut1, asserted_type=Dense)


def test_add_lr(lr1, lr2):
    with ConditionalContext(
        structured(lr1.left) or structured(lr2.left),
        AssertDenseWarning("indexing into <diagonal>"),
    ):
        check_bin_op(B.add, lr1, lr2, asserted_type=LowRank)


def test_add_lr_const(lr1, const_or_scalar2):
    with AssertDenseWarning("indexing into <diagonal>"):
        check_bin_op(B.add, const_or_scalar2, lr1, asserted_type=LowRank)
        check_bin_op(B.add, lr1, const_or_scalar2, asserted_type=LowRank)


def test_add_lr_diag(lr1, diag2):
    check_bin_op(B.add, lr1, diag2, asserted_type=Woodbury)
    check_bin_op(B.add, diag2, lr1, asserted_type=Woodbury)


def test_add_wb(wb1, wb2):
    with ConditionalContext(
        structured(wb1.lr.left) or structured(wb2.lr.left),
        AssertDenseWarning("indexing into <diagonal>"),
    ):
        check_bin_op(B.add, wb1, wb2, asserted_type=Woodbury)


def test_add_wb_diag(wb1, diag1):
    check_bin_op(B.add, wb1, diag1, asserted_type=Woodbury)
    check_bin_op(B.add, diag1, wb1, asserted_type=Woodbury)


def test_add_wb_constant(wb1, const_or_scalar2):
    with AssertDenseWarning("indexing into <diagonal>"):
        check_bin_op(B.add, wb1, const_or_scalar2, asserted_type=Woodbury)
        check_bin_op(B.add, const_or_scalar2, wb1, asserted_type=Woodbury)


def test_add_wb_lr(wb1, lr2):
    with ConditionalContext(
        structured(wb1.lr.left) or structured(lr2.left),
        AssertDenseWarning("indexing into <diagonal>"),
    ):
        check_bin_op(B.add, wb1, lr2, asserted_type=Woodbury)
        check_bin_op(B.add, lr2, wb1, asserted_type=Woodbury)


def test_kron_diag(kron1, diag1):
    with AssertDenseWarning("adding <kronecker> and <diagonal>"):
        check_bin_op(B.add, kron1, diag1, asserted_type=Dense)
