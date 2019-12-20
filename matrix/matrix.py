import abc

import lab as B
import wbml.out
from plum import Referentiable

from .shape import assert_matrix
from .util import indent, dtype_str

__all__ = ['AbstractMatrix', 'Dense']


class AbstractMatrix(metaclass=Referentiable(abc.ABCMeta)):
    """Abstract matrix type."""

    def __neg__(self):
        return B.negative(self)

    def __add__(self, other):
        return B.add(self, other)

    def __radd__(self, other):
        return B.add(other, self)

    def __sub__(self, other):
        return B.subtract(self, other)

    def __rsub__(self, other):
        return B.subtract(other, self)

    def __mul__(self, other):
        return B.multiply(self, other)

    def __rmul__(self, other):
        return B.multiply(other, self)

    def __truediv__(self, other):
        return B.divide(self, other)

    def __rtruediv__(self, other):
        return B.divide(other, self)

    def __pow__(self, power, modulo=None):
        assert modulo is None  # TODO: Implement this.
        return B.power(self, power)

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def __str__(self):  # pragma: no cover
        pass


class Dense(AbstractMatrix):
    """Dense matrix.

    Attributes:
        mat (matrix): Matrix.
        cholesky (:class:`.matrix.Dense` or None): Cholesky decomposition of
            the matrix, once it has been computed.

    Args:
        mat (matrix): Matrix.
    """

    def __init__(self, mat):
        assert_matrix(mat, 'Input is not a rank-2 tensor. Can only construct '
                           'dense matrices from rank-2 tensors.')
        self.mat = mat
        self.cholesky = None

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<dense matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' mat=' + \
               indent(wbml.out.format(self.mat, info=False),
                      ' ' * 5).strip() + '>'
