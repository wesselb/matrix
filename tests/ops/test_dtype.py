import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
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


def test_dtype_zero(zero1):
    assert B.dtype(zero1) == B.default_dtype


def test_dtype_dense(dense1):
    assert B.dtype(dense1) == B.default_dtype


def test_dtype_diag(diag1):
    assert B.dtype(diag1) == B.default_dtype


def test_dtype_const(const1):
    assert B.dtype(const1) == B.default_dtype


def test_dtype_lt1(lt1):
    assert B.dtype(lt1) == B.default_dtype


def test_dtype_ut1(ut1):
    assert B.dtype(ut1) == B.default_dtype


def test_dtype_lr(lr1):
    assert B.dtype(lr1) == B.default_dtype


def test_dtype_wb(wb1):
    assert B.dtype(wb1) == B.default_dtype


def test_dtype_kron(kron1):
    assert B.dtype(kron1) == B.default_dtype
