import lab as B
import pytest

from matrix import AbstractMatrix

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    const1,
    dense1,
    diag1,
    kron1,
    lr1,
    lt1,
    mat1,
    tb1,
    tb_axis,
    ut1,
    wb1,
    zero1,
)


def _check_cache(a):
    assert B.dense(a) is B.dense(a)


def test_dense_unknown():
    class MyMat(AbstractMatrix):
        def __str__(self):
            return "MyMat"

        def __repr__(self):
            return "MyMat"

    with pytest.raises(RuntimeError):
        B.dense(MyMat())


def test_dense_zero(zero1):
    approx(B.dense(zero1), B.zeros(zero1.rows, zero1.cols))
    _check_cache(zero1)


def test_dense_numeric(mat1):
    approx(B.dense(mat1), mat1)
    _check_cache(mat1)


def test_dense_dense(dense1):
    approx(B.dense(dense1), dense1.mat)
    _check_cache(dense1)


def test_dense_diag(diag1):
    approx(B.dense(diag1), B.diag(diag1.diag))
    _check_cache(diag1)


def test_dense_const(const1):
    approx(B.dense(const1), const1.const * B.ones(const1.rows, const1.cols))
    _check_cache(const1)


def test_dense_lt(lt1):
    approx(B.dense(lt1), lt1.mat)
    _check_cache(lt1)


def test_dense_ut(ut1):
    approx(B.dense(ut1), ut1.mat)
    _check_cache(ut1)


def test_dense_lr(lr1):
    lr_dense = B.mm(
        B.dense(lr1.left), B.dense(lr1.middle), B.dense(lr1.right), tr_c=True
    )
    approx(B.dense(lr1), lr_dense)
    _check_cache(lr1)


def test_dense_wb(wb1):
    lr_dense = B.mm(
        B.dense(wb1.lr.left), B.dense(wb1.lr.middle), B.dense(wb1.lr.right), tr_c=True
    )
    approx(B.dense(wb1), B.diag(wb1.diag.diag) + lr_dense)
    _check_cache(wb1)


def test_dense_kron(kron1):
    approx(B.dense(kron1), B.kron(B.dense(kron1.left), B.dense(kron1.right)))
    _check_cache(kron1)


def test_dense_tb(tb1):
    with AssertDenseWarning(["tiling", "concatenating"]):
        approx(
            B.dense(tb1),
            B.concat(
                *sum(
                    [
                        (B.dense(block),) * rep
                        for block, rep in zip(tb1.blocks, tb1.reps)
                    ],
                    (),
                ),
                axis=tb1.axis
            ),
        )
    _check_cache(tb1)


def test_dense_tb_axis(tb1):
    tb1.axis = 3
    with pytest.raises(RuntimeError):
        B.dense(tb1)
