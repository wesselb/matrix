import lab as B
import numpy as np

from matrix import Diagonal, Kronecker, LowRank, TiledBlocks

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
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


def test_dtype_lr_promotion():
    lr = LowRank(B.ones(int, 5, 2), B.ones(int, 5, 2), B.ones(int, 2, 2))
    assert B.dtype(lr) == np.int64
    lr = LowRank(B.ones(float, 5, 2), B.ones(int, 5, 2), B.ones(int, 2, 2))
    assert B.dtype(lr) == np.float64
    lr = LowRank(B.ones(int, 5, 2), B.ones(float, 5, 2), B.ones(int, 2, 2))
    assert B.dtype(lr) == np.float64
    lr = LowRank(B.ones(int, 5, 2), B.ones(int, 5, 2), B.ones(float, 2, 2))
    assert B.dtype(lr) == np.float64


def test_dtype_wb(wb1):
    assert B.dtype(wb1) == B.default_dtype


def test_dtype_wb_promotion():
    wb = LowRank(B.ones(int, 5, 5)) + Diagonal(B.ones(int, 5))
    assert B.dtype(wb) == np.int64
    wb = LowRank(B.ones(float, 5, 5)) + Diagonal(B.ones(int, 5))
    assert B.dtype(wb) == np.float64
    wb = LowRank(B.ones(int, 5, 5)) + Diagonal(B.ones(float, 5))
    assert B.dtype(wb) == np.float64


def test_dtype_kron(kron1):
    assert B.dtype(kron1) == B.default_dtype


def test_dtype_kron_promotion():
    kron = Kronecker(B.ones(int, 5, 5), B.ones(int, 5, 5))
    assert B.dtype(kron) == np.int64
    kron = Kronecker(B.ones(float, 5, 5), B.ones(int, 5, 5))
    assert B.dtype(kron) == np.float64
    kron = Kronecker(B.ones(int, 5, 5), B.ones(float, 5, 5))
    assert B.dtype(kron) == np.float64


def test_dtype_tb(tb1):
    assert B.dtype(tb1) == B.default_dtype


def test_dtype_tb_promotion():
    tb = TiledBlocks((B.ones(int, 5, 5), 2), (B.ones(int, 5, 5), 2))
    assert B.dtype(tb) == np.int64
    tb = TiledBlocks((B.ones(float, 5, 5), 2), (B.ones(int, 5, 5), 2))
    assert B.dtype(tb) == np.float64
    tb = TiledBlocks((B.ones(int, 5, 5), 2), (B.ones(float, 5, 5), 2))
    assert B.dtype(tb) == np.float64
