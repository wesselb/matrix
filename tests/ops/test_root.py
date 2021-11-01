import lab as B
import pytest

from matrix import Constant, Dense, Diagonal, Kronecker, LowerTriangular

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const_pd,
    dense1_pd,
    diag1_pd,
    kron_pd,
    lr1,
    lr1_pd,
    wb1_pd,
    zero1,
)


def _check_root(a, asserted_type=object):
    root = B.root(a)

    # Check correctness.
    approx(B.matmul(B.dense(root), B.dense(root)), B.dense(a))

    # Check type.
    assert isinstance(root, asserted_type)


def test_root_square_assertion():
    with pytest.raises(AssertionError):
        B.root(Dense(B.randn(3, 4)))


def test_root_zero(zero1):
    assert B.root(zero1) is zero1


def test_root_dense(dense1_pd):
    _check_root(dense1_pd, asserted_type=Dense)


def test_root_diag(diag1_pd):
    _check_root(diag1_pd, asserted_type=Diagonal)


def test_root_const(const_pd):
    _check_root(const_pd, asserted_type=Constant)


def test_root_lr(lr1_pd):
    with AssertDenseWarning("converting <low-rank> to dense"):
        _check_root(lr1_pd, asserted_type=Dense)


def test_root_wb(wb1_pd):
    with AssertDenseWarning("converting <woodbury> to dense"):
        _check_root(wb1_pd, asserted_type=Dense)


def test_root_kron(kron_pd):
    _check_root(kron_pd, asserted_type=Kronecker)
