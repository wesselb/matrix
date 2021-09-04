import lab as B

from matrix import (
    Constant,
    Dense,
    Diagonal,
    Kronecker,
    LowerTriangular,
    LowRank,
    TiledBlocks,
    UpperTriangular,
    Woodbury,
    Zero,
)

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const_r,
    dense_r,
    diag1,
    kron_r,
    lr1,
    lr_r,
    lt1,
    tb1,
    tb_axis,
    ut1,
    wb1,
    zero_r,
)


def test_transpose_zero(zero_r):
    check_un_op(B.transpose, zero_r, asserted_type=Zero)


def test_transpose_dense(dense_r):
    check_un_op(B.transpose, dense_r, asserted_type=Dense)


def test_transpose_diag(diag1):
    check_un_op(B.transpose, diag1, asserted_type=Diagonal)


def test_transpose_const(const_r):
    check_un_op(B.transpose, const_r, asserted_type=Constant)


def test_transpose_lt(lt1):
    check_un_op(B.transpose, lt1, asserted_type=UpperTriangular)


def test_transpose_ut(ut1):
    check_un_op(B.transpose, ut1, asserted_type=LowerTriangular)


def test_transpose_lr(lr_r):
    check_un_op(B.transpose, lr_r, asserted_type=LowRank)


def test_transpose_wb(wb1):
    check_un_op(B.transpose, wb1, asserted_type=Woodbury)


def test_transpose_kron(kron_r):
    check_un_op(B.transpose, kron_r, asserted_type=Kronecker)


def test_transpose_tb(tb1):
    with AssertDenseWarning(["tiling", "concatenating"]):
        check_un_op(B.transpose, tb1, asserted_type=TiledBlocks)


def test_transpose_tb_batch():
    # There will be no warning, because all matrices are already dense.
    check_un_op(
        B.transpose,
        TiledBlocks((B.randn(1, 3, 4), 2), (B.randn(2, 3, 4), 3), axis=0),
        asserted_type=TiledBlocks,
    )
