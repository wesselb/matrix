import lab as B
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

from matrix import (
    AbstractMatrix,
    Dense,
    Diagonal,
    Zero,
    Constant,
    LowRank,
    Woodbury
)

__all__ = ['allclose',
           'approx',
           'check_un_op',
           'check_bin_op',

           # Fixtures:
           'mat1',
           'mat2',
           'vec1',
           'vec2',
           'scalar1',
           'scalar2',

           'zero1',
           'zero2',
           'dense1',
           'dense2',
           'diag1',
           'diag2',
           'const1',
           'const2',
           'const_or_scalar1',
           'const_or_scalar2',
           'lr1',
           'lr2',
           'wb1',
           'wb2']

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


def _assert_instance(x, asserted_type):
    assert isinstance(x, asserted_type), \
        f'Expected instance of type {asserted_type} but got {type(x)}.'


def check_un_op(op, x, asserted_type=object):
    """Assert the correct of a unary operation by checking whether the
    result is the same on the dense version of the argument.

    Args:
        op (function): Unary operation to check.
        x (object): Argument.
        asserted_type (type, optional): Type of result.
    """
    x_dense = B.dense(x)
    res = op(x)
    allclose(res, op(x_dense))
    _assert_instance(res, asserted_type)


def check_bin_op(op, x, y, asserted_type=object):
    """Assert the correct of a binary operation by checking whether the
    result is the same on the dense versions of the arguments.

    Args:
        op (function): Binary operation to check.
        x (object): First argument.
        y (object): Second argument.
        asserted_type (type, optional): Type of result.
    """
    x_dense = B.dense(x)
    y_dense = B.dense(y)
    res = op(x, y)

    allclose(res, op(x_dense, y_dense))
    allclose(op(x_dense, y), op(x_dense, y_dense))
    allclose(op(x, y_dense), op(x_dense, y_dense))
    allclose(op(x_dense, y_dense), op(x_dense, y_dense))

    _assert_instance(res, asserted_type)


# Fixtures:

@pytest.fixture()
def mat1():
    return B.randn(3, 3)


@pytest.fixture()
def mat2():
    return B.randn(3, 3)


@pytest.fixture()
def vec1():
    yield B.randn(3)


@pytest.fixture()
def vec2():
    yield B.randn(3)


@pytest.fixture()
def scalar1():
    yield B.randn()


@pytest.fixture()
def scalar2():
    yield B.randn()


@pytest.fixture()
def zero1():
    yield Zero(3, 3)


@pytest.fixture()
def zero2():
    yield Zero(3, 3)


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


@pytest.fixture()
def const1():
    yield Constant(B.randn(), 3, 3)


@pytest.fixture()
def const2():
    yield Constant(B.randn(), 3, 3)


@pytest.fixture(params=[True, False])
def const_or_scalar1(request):
    if request.param:
        yield B.randn()
    else:
        yield Constant(B.randn(), 3, 3)


@pytest.fixture(params=[True, False])
def const_or_scalar2(request):
    if request.param:
        yield B.randn()
    else:
        yield Constant(B.randn(), 3, 3)


@pytest.fixture(params=[1, 2, 3])
def lr1(request):
    yield LowRank(B.randn(3, request.param), B.randn(3, request.param))


@pytest.fixture(params=[1, 2, 3])
def lr2(request):
    yield LowRank(B.randn(3, request.param), B.randn(3, request.param))


@pytest.fixture(params=[1, 2, 3])
def wb1(request):
    d = Diagonal(B.randn(3))
    lr = LowRank(B.randn(3, request.param), B.randn(3, request.param))
    return Woodbury(d, lr)


@pytest.fixture(params=[1, 2, 3])
def wb2(request):
    d = Diagonal(B.randn(3))
    lr = LowRank(B.randn(3, request.param), B.randn(3, request.param))
    return Woodbury(d, lr)
