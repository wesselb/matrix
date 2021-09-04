import lab as B

from matrix import Dense, Diagonal, structured

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    ConditionalContext,
    approx,
    check_un_op,
    concat_warnings,
    const1,
    dense1,
    dense2,
    diag1,
    diag2,
    kron1,
    lr1,
    lt1,
    ut1,
    wb1,
    zero1,
)


def test_diag_zero(zero1):
    check_un_op(B.diag, zero1, ref=B.diag_extract)


def test_diag_dense(dense1):
    check_un_op(B.diag, dense1, ref=B.diag_extract)


def test_diag_diag(diag1):
    check_un_op(B.diag, diag1, ref=B.diag_extract)


def test_diag_const(const1):
    check_un_op(B.diag, const1, ref=B.diag_extract)


def test_diag_lt1(lt1):
    check_un_op(B.diag, lt1, ref=B.diag_extract)


def test_diag_ut1(ut1):
    check_un_op(B.diag, ut1, ref=B.diag_extract)


def test_diag_lr(lr1):
    warn = AssertDenseWarning("getting the diagonal of <low-rank>")
    with ConditionalContext(structured(lr1.left, lr1.right), warn):
        check_un_op(B.diag, lr1, ref=B.diag_extract)


def test_diag_wb(wb1):
    warn = AssertDenseWarning("getting the diagonal of <low-rank>")
    with ConditionalContext(structured(wb1.lr.left, wb1.lr.right), warn):
        check_un_op(B.diag, wb1, ref=B.diag_extract)


def test_diag_kron(kron1):
    check_un_op(B.diag, kron1, ref=B.diag_extract)
