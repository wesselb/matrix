import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_vector
from .util import indent, dtype_str

__all__ = ['Diagonal']


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Attributes:
        diag (vector): Diagonal of the matrix.
        cholesky (:class:`.constant.Diagonal` or none): Cholesky
            decomposition of the matrix, once it has been computed.

    Args:
        diag (vector): Diagonal of matrix.
    """

    def __init__(self, diag):
        assert_vector(diag, 'Input is not a rank-1 tensor. Can only construct '
                            'diagonal matrices from rank-1 tensors.')
        self.diag = diag
        self.cholesky = None

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<diagonal matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' diag=' + \
               indent(wbml.out.format(self.diag, info=False),
                      ' ' * 6).strip() + '>'
