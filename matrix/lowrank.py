import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_matrix
from .util import indent, dtype_str

__all__ = ['LowRank']


class LowRank(AbstractMatrix):
    """Low-rank matrix.

    The data type of a low-rank matrix is the data type of the left factor.

    Args:
        left (matrix): Left factor.
        right (matrix, optional): Right factor. Defaults to the left factor.
    """

    def __init__(self, left, right=None):
        for factor in [left, right]:
            assert_matrix(factor, 'Factors are not rank-2 tensors. Can only '
                                  'construct low-rank matrices from matrix '
                                  'factors.')
        self.left = left
        self.right = left if right is None else right

        if not B.shape(self.left)[1] == B.shape(self.right)[1]:
            raise ValueError('Left and right factor must have an equal number '
                             'of columns.')

    def __str__(self):
        rows, cols = B.shape(self)
        rank = B.shape(self.left)[1]
        return f'<low-rank matrix:' \
               f' shape={rows}x{cols},' \
               f' data type={dtype_str(self)},' + \
               f' rank={rank},\n' + \
               f' left=' + \
               indent(wbml.out.format(self.left, info=False),
                      ' ' * 6).strip() + ',\n' + \
               ' right=' + \
               indent(wbml.out.format(self.right, info=False),
                      ' ' * 7).strip() + '>'
