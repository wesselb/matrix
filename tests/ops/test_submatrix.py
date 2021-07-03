import jax.numpy as jnp
import lab.jax as B
import numpy as np
import pytest

from matrix import (
    structured,
    Constant,
    Dense,
    Diagonal,
    Kronecker,
    LowerTriangular,
    LowRank,
    UpperTriangular,
    Woodbury,
    Zero,
)

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    ConditionalContext,
    approx,
    check_un_op,
    const1,
    dense1,
    diag1,
    kron1,
    lr1,
    lt1,
    mat1,
    ut1,
    wb1,
    zero1,
)


def _check_submatrix(a, asserted_type=object):
    for indices_or_mask in [
        [0, 1],
        [0, 2],
        [True, True, False, False, True, False],
    ]:
        for wrap in [list, tuple, np.array]:
            res = B.submatrix(a, wrap(indices_or_mask))
            approx(res, B.submatrix(B.dense(a), wrap(indices_or_mask)))
            assert isinstance(res, asserted_type)


def test_submatrix_zero(zero1):
    with pytest.raises(ValueError):
        B.submatrix(zero1, [[1]])
    _check_submatrix(zero1, asserted_type=Zero)


def test_submatrix_dense(dense1):
    _check_submatrix(dense1, asserted_type=Dense)


def test_submatrix_diag(diag1):
    _check_submatrix(diag1, asserted_type=Diagonal)


def test_submatrix_const(const1):
    with pytest.raises(ValueError):
        B.submatrix(const1, [[1]])
    _check_submatrix(const1, asserted_type=Constant)


def test_submatrix_lt(lt1):
    _check_submatrix(lt1, asserted_type=LowerTriangular)


def test_submatrix_ut(ut1):
    _check_submatrix(ut1, asserted_type=UpperTriangular)


warn = AssertDenseWarning("taking from <diagonal>")


def test_submatrix_lr(lr1):
    with ConditionalContext(structured(lr1.left, lr1.right), warn):
        _check_submatrix(lr1, asserted_type=LowRank)


def test_submatrix_wb(wb1):
    with ConditionalContext(structured(wb1.lr.left, wb1.lr.right), warn):
        _check_submatrix(wb1, asserted_type=Woodbury)


def test_submatrix_kron(kron1):
    with AssertDenseWarning("taking a submatrix from <kronecker>"):
        _check_submatrix(kron1, asserted_type=Dense)
