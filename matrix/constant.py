import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_scalar
from .util import dtype_str

__all__ = ['Constant', 'Zero']


class Constant(AbstractMatrix):
    """Constant matrix.

    Args:
        const (scalar): Constant.
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, const, rows, cols):
        assert_scalar(const, 'Input is not a scalar. Can only construct '
                             'constant matrices from scalars.')
        self.const = const
        self.rows = rows
        self.cols = cols

    def __str__(self):
        rows, cols = B.shape(self)
        return f'Constant {rows}x{cols} matrix ' + \
               f'of data type {dtype_str(self)} ' + \
               f'with constant ' + wbml.out.format(self.const, info=False)


class Zero(Constant):
    """Zero matrix.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, rows, cols):
        Constant.__init__(self, 0.0, rows, cols)

    def __str__(self):
        rows, cols = B.shape(self)
        return f'Constant {rows}x{cols} matrix of zeros'