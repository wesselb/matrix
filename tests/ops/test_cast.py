import lab as B
import numpy as np

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


def check_casting(x):
    assert B.default_dtype != complex
    assert B.dtype(B.cast(np.int64, x)) == np.int64
    with IgnoreDenseWarning():
        check_un_op(lambda x: B.cast(complex, x), x)


def test_dtype_zero(zero1):
    check_casting(zero1)


def test_dtype_dense(dense1):
    check_casting(dense1)


def test_dtype_diag(diag1):
    check_casting(diag1)


def test_dtype_const(const1):
    check_casting(const1)


def test_dtype_lt1(lt1):
    check_casting(lt1)


def test_dtype_ut1(ut1):
    check_casting(ut1)


def test_dtype_lr(lr1):
    check_casting(lr1)


def test_dtype_wb(wb1):
    check_casting(wb1)


def test_dtype_kron(kron1):
    check_casting(kron1)


def test_dtype_tb(tb1):
    check_casting(tb1)
