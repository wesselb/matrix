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
    Woodbury,
    Kronecker
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
           'zero_r',
           'dense1',
           'dense2',
           'dense_r',
           'dense_pd',
           'diag1',
           'diag2',
           'diag_pd',
           'const1',
           'const2',
           'const_r',
           'const_pd',
           'const_or_scalar1',
           'const_or_scalar2',
           'lr1',
           'lr2',
           'lr_r',
           'lr_pd',
           'wb1',
           'wb2',
           'kron1',
           'kron2',
           'kron_r',
           'kron_mixed',
           'kron_pd']

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


@_dispatch({B.Number, B.NPNumeric}, {B.Number, B.NPNumeric})
def allclose(x, y):
    assert_allclose(x, y, rtol=1e-7, atol=1e-14)


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
    return B.randn(6, 6)


@pytest.fixture()
def mat2():
    return B.randn(6, 6)


@pytest.fixture()
def vec1():
    yield B.randn(6)


@pytest.fixture()
def vec2():
    yield B.randn(6)


@pytest.fixture()
def scalar1():
    yield B.randn()


@pytest.fixture()
def scalar2():
    yield B.randn()


@pytest.fixture()
def zero1():
    yield Zero(float, 6, 6)


@pytest.fixture()
def zero2():
    yield Zero(float, 6, 6)


@pytest.fixture()
def zero_r():
    yield Zero(float, 6, 4)


@pytest.fixture()
def dense1():
    yield Dense(B.randn(6, 6))


@pytest.fixture()
def dense2():
    yield Dense(B.randn(6, 6))


@pytest.fixture()
def dense_r():
    yield Dense(B.randn(6, 4))


@pytest.fixture()
def dense_pd():
    mat = B.randn(6, 6)
    yield Dense(B.matmul(mat, mat, tr_b=True))


@pytest.fixture()
def diag1():
    yield Diagonal(B.randn(6))


@pytest.fixture()
def diag2():
    yield Diagonal(B.randn(6))


@pytest.fixture()
def diag_pd():
    yield Diagonal(B.randn(6) ** 2)


@pytest.fixture()
def const1():
    yield Constant(B.randn(), 6, 6)


@pytest.fixture()
def const2():
    yield Constant(B.randn(), 6, 6)


@pytest.fixture()
def const_r():
    yield Constant(B.randn(), 6, 6)


@pytest.fixture()
def const_pd():
    yield Constant(B.randn() ** 2, 6, 6)


@pytest.fixture(params=[True, False])
def const_or_scalar1(request):
    if request.param:
        yield B.randn()
    else:
        yield Constant(B.randn(), 6, 6)


@pytest.fixture(params=[True, False])
def const_or_scalar2(request):
    if request.param:
        yield B.randn()
    else:
        yield Constant(B.randn(), 6, 6)


@pytest.fixture(params=[1, 2, 3])
def lr1(request):
    yield LowRank(B.randn(6, request.param), B.randn(6, request.param))


@pytest.fixture(params=[1, 2, 3])
def lr2(request):
    yield LowRank(B.randn(6, request.param), B.randn(6, request.param))


@pytest.fixture(params=[1, 2, 3])
def lr_r(request):
    yield LowRank(B.randn(6, request.param), B.randn(4, request.param))


@pytest.fixture(params=[1, 2, 3])
def lr_pd(request):
    yield LowRank(B.randn(6, request.param))


@pytest.fixture(params=[1, 2, 3])
def wb1(request):
    d = Diagonal(B.randn(6))
    lr = LowRank(B.randn(6, request.param), B.randn(6, request.param))
    return Woodbury(d, lr)


@pytest.fixture(params=[1, 2, 3])
def wb2(request):
    d = Diagonal(B.randn(6))
    lr = LowRank(B.randn(6, request.param), B.randn(6, request.param))
    return Woodbury(d, lr)


@pytest.fixture(params=[(1, 6), (2, 3), (3, 2), (6, 1)])
def kron1(request):
    size_left, size_right = request.param
    left = B.randn(size_left, size_left)
    right = B.randn(size_right, size_right)
    return Kronecker(Dense(left), Dense(right))


@pytest.fixture(params=[(1, 6), (2, 3), (3, 2), (6, 1)])
def kron2(request):
    size_left, size_right = request.param
    left = B.randn(size_left, size_left)
    right = B.randn(size_right, size_right)
    return Kronecker(Dense(left), Dense(right))


@pytest.fixture(params=[(1, 6), (2, 3), (3, 2), (6, 1)])
def kron_r(request):
    size_left, size_right = request.param
    left = B.randn(size_left, 2)
    right = B.randn(size_right, 2)
    return Kronecker(Dense(left), Dense(right))


@pytest.fixture(params=[((1, 6), (6, 1)),
                        ((2, 3), (3, 2)),
                        ((3, 2), (2, 3)),
                        ((6, 1), (1, 6))])
def kron_mixed(request):
    sizes_left, sizes_right = request.param
    left = B.randn(*sizes_left)
    right = B.randn(*sizes_right)
    return Kronecker(Dense(left), Dense(right))


@pytest.fixture(params=[(1, 6), (2, 3), (3, 2), (6, 1)])
def kron_pd(request):
    size_left, size_right = request.param
    left = B.randn(size_left, size_left)
    right = B.randn(size_right, size_right)
    return Kronecker(Dense(B.matmul(left, left, tr_b=True)),
                     Dense(B.matmul(right, right, tr_b=True)))
