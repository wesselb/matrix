import lab as B
import wbml.out
from .util import dtype_str
from .shape import assert_scalar

from .matrix import AbstractMatrix


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
