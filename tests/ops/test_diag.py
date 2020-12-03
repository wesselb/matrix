import lab as B

from matrix import Dense, Diagonal, structured

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    ConditionalContext,
    concat_warnings,
    zero1,
    dense1,
    dense2,
    diag1,
    diag2,
    const1,
    lt1,
    ut1,
    lr1,
    wb1,
    kron1,
)


def test_diag_zero(zero1):
    check_un_op(B.diag, zero1)


def test_diag_dense(dense1):
    check_un_op(B.diag, dense1)


def test_diag_diag(diag1):
    check_un_op(B.diag, diag1)


def test_diag_const(const1):
    check_un_op(B.diag, const1)


def test_diag_lt1(lt1):
    check_un_op(B.diag, lt1)


def test_diag_ut1(ut1):
    check_un_op(B.diag, ut1)


def test_diag_lr(lr1):
    warn = AssertDenseWarning("getting the diagonal of <low-rank>")
    with ConditionalContext(structured(lr1.left, lr1.right), warn):
        check_un_op(B.diag, lr1)


def test_diag_wb(wb1):
    warn = AssertDenseWarning("getting the diagonal of <low-rank>")
    with ConditionalContext(structured(wb1.lr.left, wb1.lr.right), warn):
        check_un_op(B.diag, wb1)


def test_diag_kron(kron1):
    check_un_op(B.diag, kron1)


def test_diag_block_dense(dense1, dense2):
    with AssertDenseWarning(concat_warnings):
        res = B.diag(dense1, dense2)
        approx(
            res,
            B.concat2d(
                [B.dense(dense1), B.zeros(B.dense(dense1))],
                [B.zeros(B.dense(dense2)), B.dense(dense2)],
            ),
        )
        assert isinstance(res, Dense)


def test_diag_block_diag(diag1, diag2):
    approx(
        B.diag(diag1, diag2),
        B.concat2d(
            [B.dense(diag1), B.zeros(B.dense(diag2))],
            [B.zeros(B.dense(diag2)), B.dense(diag2)],
        ),
    )
    assert isinstance(B.diag(diag1, diag2), Diagonal)
