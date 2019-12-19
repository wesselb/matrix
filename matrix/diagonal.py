import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_vector
from .util import indent, dtype_str

__all__ = ['Diagonal']


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Args:
        diag (vector): Diagonal of matrix.
    """

    def __init__(self, diag):
        assert_vector(diag, 'Input is not a rank-1 tensor. Can only construct '
                            'diagonal matrices from rank-1 tensors.')
        self.diag = diag

    def __str__(self):
        rows, cols = B.shape(self)
        return f'Diagonal {rows}x{cols} matrix ' + \
               f'of data type {dtype_str(self)} ' + \
               f'with diagonal\n' + \
               indent(wbml.out.format(self.diag, info=False))
