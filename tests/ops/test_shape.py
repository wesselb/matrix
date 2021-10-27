import lab as B
import pytest
from lab.shape import Shape

from matrix.ops.shape import _drop_axis

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


def test_shape_broadcast():
    assert B.shape_broadcast(B.ones(3, 1, 1), B.ones(4, 1), B.ones(1, 5)) == (3, 4, 5)
    assert B.shape_batch_broadcast(B.ones(3, 1, 1), B.ones(4, 1), B.ones(1, 5)) == (3,)
    shape = B.shape_matrix_broadcast(B.ones(3, 1, 1), B.ones(4, 1), B.ones(1, 5))
    assert shape == (4, 5)


def test_shape_indexing():
    assert B.shape_batch(B.ones(3, 4, 5, 6), 0) == 3
    assert B.shape_batch(B.ones(3, 4, 5, 6), 0, 1) == (3, 4)
    assert B.shape_matrix(B.ones(3, 4, 5, 6), 0) == 5
    assert B.shape_matrix(B.ones(3, 4, 5, 6), 0, 1) == (5, 6)


@pytest.fixture(params=[B.shape, B.shape_batch, B.shape_matrix])
def shape(request):
    return request.param


def test_shape_zero(shape, zero_r):
    check_un_op(shape, zero_r, asserted_type=(tuple, Shape))


def test_shape_dense(shape, dense_r):
    check_un_op(shape, dense_r, asserted_type=(tuple, Shape))


def test_shape_diag(shape, diag1):
    check_un_op(shape, diag1, asserted_type=(tuple, Shape))


def test_shape_const(shape, const_r):
    check_un_op(shape, const_r, asserted_type=(tuple, Shape))


def test_shape_lt(shape, lt1):
    check_un_op(shape, lt1, asserted_type=(tuple, Shape))


def test_shape_ut(shape, ut1):
    check_un_op(shape, ut1, asserted_type=(tuple, Shape))


def test_shape_lr(shape, lr_r):
    check_un_op(shape, lr_r, asserted_type=(tuple, Shape))


def test_shape_wb(shape, wb1):
    check_un_op(shape, wb1, asserted_type=(tuple, Shape))


def test_shape_kron(shape, kron_r):
    check_un_op(shape, kron_r, asserted_type=(tuple, Shape))


def test_drop_axis():
    assert _drop_axis((1, 2, 3), -3) == (2, 3)
    assert _drop_axis((1, 2, 3), -2) == (1, 3)
    assert _drop_axis((1, 2, 3), -1) == (1, 2)
    assert _drop_axis((1, 2, 3), 0) == (2, 3)
    assert _drop_axis((1, 2, 3), 1) == (1, 3)
    assert _drop_axis((1, 2, 3), 2) == (1, 2)


def test_shape_tb(shape, tb1):
    with AssertDenseWarning(["tiling", "concatenating"]):
        check_un_op(shape, tb1, asserted_type=(tuple, Shape))
