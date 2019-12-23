import re

import warnings
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
from matrix.util import ToDenseWarning

__all__ = ['allclose',
           'approx',
           'check_un_op',
           'check_bin_op',
           'AssertDenseWarning',

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
           'dense_bc',
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
           'wb_pd',
           'kron1',
           'kron2',
           'kron_r',
           'kron_pd'
           'kron_mixed']

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

    warnings.filterwarnings(category=ToDenseWarning, action='ignore')
    allclose(op(x_dense, y), op(x_dense, y_dense))
    allclose(op(x, y_dense), op(x_dense, y_dense))
    allclose(op(x_dense, y_dense), op(x_dense, y_dense))
    warnings.filterwarnings(category=ToDenseWarning, action='default')

    _assert_instance(res, asserted_type)


def _sanitise(msg):
    # Filter details from printed objects
    msg = re.sub(r'<([a-zA-Z\- ]{1,}):[^>]*>', r'<\1>', msg)
    return msg


class AssertDenseWarning:
    """Assert that a `ToDenseWarning` is raised with a particular content.

    Args:
        content (str): Content that the arguments of the warning must contain.
    """

    def __init__(self, content):
        self.content = content.lower()
        self.record = None

    def __enter__(self):
        self.context = pytest.warns(ToDenseWarning)
        self.record = self.context.__enter__()
        return self.record

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.__exit__(exc_type, exc_val, exc_tb)

        # Perform assertions.
        for i in range(len(self.record)):
            message = ''.join(self.record[i].message.args)
            message = _sanitise(message).lower()

            if self.content not in message:
                raise AssertionError(f'Warning should contain '
                                     f'"{self.content}", but it did not: '
                                     f'"{message}".')


def generate(code):
    """Generate a random tensor of a particular type, specified with a code.

    Args:
        code (str): Code of the matrix.

    Returns:
        tensor: Random tensor.
    """
    mat_code, shape_code = code.split(':')

    # Parse shape.
    if shape_code == '':
        shape = ()
    else:
        shape = tuple(int(d) for d in shape_code.split(','))

    if mat_code == 'randn':
        return B.randn(*shape)
    elif mat_code == 'randn_pd':
        mat = B.randn(*shape)

        # If it is a scalar or vector, just pointwise square it.
        if len(shape) in {0, 1}:
            return mat ** 2
        else:
            return B.matmul(mat, mat, tr_b=True)

    elif mat_code == 'zero':
        return Zero(float, *shape)

    elif mat_code == 'const':
        return Constant(B.randn(), *shape)
    elif mat_code == 'const_pd':
        return Constant(B.randn() ** 2, *shape)

    elif mat_code == 'dense':
        return Dense(generate(f'randn:{shape_code}'))
    elif mat_code == 'dense_pd':
        return Dense(generate(f'randn_pd:{shape_code}'))

    elif mat_code == 'diag':
        return Diagonal(generate(f'randn:{shape_code}'))
    elif mat_code == 'diag_pd':
        return Diagonal(generate(f'randn_pd:{shape_code}'))

    else:
        raise RuntimeError(f'Cannot parse generation code "{code}".')


# Fixtures:

@pytest.fixture()
def mat1():
    return generate('randn:6,6')


@pytest.fixture()
def mat2(mat1):
    return generate('randn:6,6')


@pytest.fixture()
def vec1():
    return generate('randn:6')


@pytest.fixture()
def vec2():
    return generate('randn:6')


@pytest.fixture()
def scalar1():
    return generate('randn:')


@pytest.fixture()
def scalar2():
    return generate('randn:')


@pytest.fixture()
def zero1():
    return generate('zero:6,6')


@pytest.fixture()
def zero2():
    return generate('zero:6,6')


@pytest.fixture()
def zero_r():
    return generate('zero:6,4')


@pytest.fixture()
def dense1():
    return generate('dense:6,6')


@pytest.fixture()
def dense2():
    return generate('dense:6,6')


@pytest.fixture(params=(['dense:6,1', 'dense:1,6', 'dense:6,6']))
def dense_bc(request):
    return generate(request.param)


@pytest.fixture()
def dense_r():
    return generate('dense:6,4')


@pytest.fixture()
def dense_pd():
    return generate('dense_pd:6,6')


@pytest.fixture()
def diag1():
    return generate('diag:6')


@pytest.fixture()
def diag2():
    return generate('diag:6')


@pytest.fixture()
def diag_pd():
    return generate('diag_pd:6')


@pytest.fixture()
def const1():
    return generate('const:6,6')


@pytest.fixture()
def const2():
    return generate('const:6,6')


@pytest.fixture()
def const_r():
    return generate('const:6,4')


@pytest.fixture()
def const_pd():
    return generate('const_pd:6,6')


@pytest.fixture(params=['randn:', 'const:6,6'])
def const_or_scalar1(request):
    return generate(request.param)


@pytest.fixture(params=['randn:', 'const:6,6'])
def const_or_scalar2(request):
    return generate(request.param)


@pytest.fixture(params=[('dense:6,1', 'dense:6,1'),
                        ('dense:6,2', 'dense:6,2'),
                        ('dense:6,3', 'dense:6,3'),
                        ('diag:6', 'diag:6')])
def lr1(request):
    code1, code2 = request.param
    return LowRank(generate(code1), generate(code2))


@pytest.fixture(params=[('dense:6,1', 'dense:6,1'),
                        ('dense:6,2', 'dense:6,2'),
                        ('dense:6,3', 'dense:6,3'),
                        ('diag:6', 'diag:6')])
def lr2(request):
    code1, code2 = request.param
    return LowRank(generate(code1), generate(code2))


@pytest.fixture(params=[('dense:6,1', 'dense:4,1'),
                        ('dense:6,2', 'dense:4,2'),
                        ('dense:6,3', 'dense:4,3'),
                        ('diag:6', 'dense:4,6')])
def lr_r(request):
    code1, code2 = request.param
    return LowRank(generate(code1), generate(code2))


@pytest.fixture(params=['dense:6,1', 'dense:6,2', 'dense:6,3', 'diag:6'])
def lr_pd(request):
    return LowRank(generate(request.param))


@pytest.fixture()
def wb1(diag1, lr1):
    return Woodbury(diag1, lr1)


@pytest.fixture()
def wb2(diag2, lr2):
    return Woodbury(diag2, lr2)


@pytest.fixture()
def wb_pd(diag_pd, lr_pd):
    return Woodbury(diag_pd, lr_pd)


@pytest.fixture(params=[('diag:1', 'diag:6'),
                        ('diag:2', 'diag:3'),
                        ('diag:3', 'diag:2'),
                        ('diag:6', 'diag:1'),
                        ('dense:1,1', 'dense:6,6'),
                        ('dense:2,2', 'dense:3,3'),
                        ('dense:3,3', 'dense:2,2'),
                        ('dense:6,6', 'dense:1,1')])
def kron1(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(params=[('diag:1', 'diag:6'),
                        ('diag:2', 'diag:3'),
                        ('diag:3', 'diag:2'),
                        ('diag:6', 'diag:1'),
                        ('dense:1,1', 'dense:1,1'),
                        ('dense:2,2', 'dense:3,3'),
                        ('dense:3,3', 'dense:2,2'),
                        ('dense:6,6', 'dense:1,1')])
def kron2(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(params=[('diag:2', 'dense:3,2'),
                        ('dense:3,2', 'diag:2'),
                        ('dense:1,4', 'dense:6,1'),
                        ('dense:6,1', 'dense:1,4')])
def kron_r(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(params=[('diag_pd:1', 'diag_pd:6'),
                        ('diag_pd:2', 'diag_pd:3'),
                        ('diag_pd:3', 'diag_pd:2'),
                        ('diag_pd:6', 'diag_pd:1'),
                        ('dense_pd:1,1', 'dense_pd:6,6'),
                        ('dense_pd:2,2', 'dense_pd:3,3'),
                        ('dense_pd:3,3', 'dense_pd:2,2'),
                        ('dense_pd:6,6', 'dense_pd:1,1')])
def kron_pd(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(params=[('diag:1', 'diag:6'),
                        ('diag:2', 'diag:3'),
                        ('diag:3', 'diag:2'),
                        ('diag:6', 'diag:1'),
                        ('dense:1,6', 'dense:6,1'),
                        ('dense:2,3', 'dense:3,2'),
                        ('dense:3,2', 'dense:2,3'),
                        ('dense:6,1', 'dense:1,6')])
def kron_mixed(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))
