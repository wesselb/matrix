import lab as B
import numpy as np
from matrix import (
    Zero,
    Dense,
    Diagonal,
    Constant,
    LowerTriangular,
    UpperTriangular,
    LowRank,
    Woodbury,
    Kronecker,
    TiledBlocks,
)

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    IgnoreDenseWarning,
    const1,
    dense1,
    diag1,
    kron1,
    lr1,
    lt1,
    tb1,
    tb_axis,
    ut1,
    wb1,
    zero1,
)


def check_casting(x, asserted_type):
    assert B.default_dtype != complex
    assert B.dtype(B.cast(np.int64, x)) == np.int64
    with IgnoreDenseWarning():
        check_un_op(lambda x: B.cast(complex, x), x, asserted_type=asserted_type)


def test_cast_zero(zero1):
    check_casting(zero1, asserted_type=Zero)


def test_cast_dense(dense1):
    check_casting(dense1, asserted_type=Dense)


def test_cast_diag(diag1):
    check_casting(diag1, asserted_type=Diagonal)


def test_cast_const(const1):
    check_casting(const1, asserted_type=Constant)


def test_cast_lt1(lt1):
    check_casting(lt1, asserted_type=LowerTriangular)


def test_cast_ut1(ut1):
    check_casting(ut1, asserted_type=UpperTriangular)


def test_cast_lr(lr1):
    check_casting(lr1, asserted_type=LowRank)


def test_cast_wb(wb1):
    check_casting(wb1, asserted_type=Woodbury)


def test_cast_kron(kron1):
    check_casting(kron1, asserted_type=Kronecker)


def test_cast_tb(tb1):
    check_casting(tb1, asserted_type=TiledBlocks)
