import lab as B
import wbml.out
from plum import Dispatcher, Self

from .matrix import AbstractMatrix
from .util import indent, dtype_str

__all__ = ['LowerTriangular', 'UpperTriangular']


class LowerTriangular(AbstractMatrix):
    """Lower-triangular matrix.

    Attributes:
        mat (matrix): Dense lower-triangular matrix.

    Args:
        mat (matrix): Dense lower-triangular matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, mat):
        self.mat = mat

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<lower-triangular matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' mat=' + indent(wbml.out.format(self.mat, info=False),
                                 ' ' * 5).strip() + '>'


class UpperTriangular(AbstractMatrix):
    """Upper-triangular matrix.

    Attributes:
        mat (matrix): Dense upper-triangular matrix.

    Args:
        mat (matrix): Dense upper-triangular matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, mat):
        self.mat = mat

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<lower-triangular matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' mat=' + indent(wbml.out.format(self.mat, info=False),
                                 ' ' * 5).strip() + '>'
