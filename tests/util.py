import lab as B
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

from matrix import AbstractMatrix, Dense, Diagonal

__all__ = ['allclose',
           'approx',
           'check_un_op',
           'check_bin_op',

           # Fixtures:
           'mat1',
           'vec1',
           'vec2',
           'dense1',
           'dense2',
           'diag1',
           'diag2']

_dispatch = Dispatcher()

approx = assert_array_almost_equal

# Convert structured matrices to numpy by constructing their dense versions.
B.to_numpy.extend(AbstractMatrix)(B.dense)


@_dispatch(object, object)
def allclose(x, y):
    """Assert that two objects are numerically close.

    Args:
        x (object): First object.
        y (object): Second object.
    """
    allclose(B.to_numpy(x), B.to_numpy(y))


@_dispatch(B.NPNumeric, B.NPNumeric)
def allclose(x, y):
    assert_allclose(x, y)


def check_un_op(op, x, asserted_type):
    """Assert the correct of a unary operation by checking whether the
    result is the same on the dense version of the argument.

    Args:
        op (function): Unary operation to check.
        x (object): Argument.
        asserted_type (type): Type of result.
    """
    x_dense = B.dense(x)
    allclose(op(x), op(x_dense))
    assert isinstance(op(x), asserted_type)


def check_bin_op(op, x, y, asserted_type):
    """Assert the correct of a binary operation by checking whether the
    result is the same on the dense versions of the arguments.

    Args:
        op (function): Binary operation to check.
        x (object): First argument.
        y (object): Second argument.
        asserted_type (type): Type of result.
    """
    x_dense = B.dense(x)
    y_dense = B.dense(y)

    allclose(op(x, y), op(x_dense, y_dense))
    allclose(op(x_dense, y), op(x_dense, y_dense))
    allclose(op(x, y_dense), op(x_dense, y_dense))
    allclose(op(x_dense, y_dense), op(x_dense, y_dense))

    assert isinstance(op(x, y), asserted_type)


# Fixtures:

@pytest.fixture()
def mat1():
    return B.randn(3, 3)


@pytest.fixture()
def vec1():
    yield B.randn(3)


@pytest.fixture()
def vec2():
    yield B.randn(3)


@pytest.fixture()
def dense1():
    yield Dense(B.randn(3, 3))


@pytest.fixture()
def dense2():
    yield Dense(B.randn(3, 3))


@pytest.fixture()
def diag1():
    yield Diagonal(B.randn(3))


@pytest.fixture()
def diag2():
    yield Diagonal(B.randn(3))
