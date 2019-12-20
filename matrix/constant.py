import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_scalar
from .util import dtype_str

__all__ = ['Constant', 'Zero']


class Constant(AbstractMatrix):
    """Constant matrix.

    Attributes:
        const (scalar): Constant of the matrix.
        rows (int): Number of rows.
        cols (int): Number of columns.
        cholesky (:class:`.constant.Constant` or None): Cholesky
            decomposition of the matrix, once it has been computed.

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
        self.cholesky = None

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<constant matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},' + \
               f' const=' + wbml.out.format(self.const, info=False) + '>'


class Zero(Constant):
    """Zero matrix.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, rows, cols):
        Constant.__init__(self, 0, rows, cols)

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<zero matrix: shape={rows}x{cols}>'
