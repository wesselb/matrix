import lab as B
import wbml.out

from .matrix import AbstractMatrix
from .shape import assert_matrix
from .util import indent, dtype_str

__all__ = ['LowRank']


class LowRank(AbstractMatrix):
    """Low-rank matrix.

    The data type of a low-rank matrix is the data type of the left factor.
    The Cholesky decomposition is not exactly the Cholesky decomposition,
    but returns a matrix `L` such that `L transpose(L)` is the original martix.

    Attributes:
        left (matrix): Left factor.
        right (matrix): Right factor.
        rank (int): Rank of the low-rank matrix.
        symmetric (bool): Boolean indicating whether this is a symmetric
            low-rank matrix.
        cholesky (:class:`.matrix.Dense` or None): Cholesky-like
            decomposition of the matrix, once it has been computed.

    Args:
        left (matrix): Left factor.
        right (matrix, optional): Right factor. Defaults to the left factor.
    """

    def __init__(self, left, right=None):
        if right is None:
            check_factors = [left]
        else:
            check_factors = [left, right]
        for factor in check_factors:
            assert_matrix(factor, 'Factors are not rank-2 tensors. Can only '
                                  'construct low-rank matrices from matrix '
                                  'factors.')
        self.left = left
        self.right = left if right is None else right
        self.rank = B.shape(self.left)[1]
        self.symmetric = right is None
        self.cholesky = None

        if not B.shape(self.left)[1] == B.shape(self.right)[1]:
            raise ValueError('Left and right factor must have an equal number '
                             'of columns.')

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<low-rank matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},' + \
               f' rank={self.rank},\n' + \
               f' left=' + \
               indent(wbml.out.format(self.left, info=False),
                      ' ' * 6).strip() + ',\n' + \
               f' right=' + \
               indent(wbml.out.format(self.right, info=False),
                      ' ' * 7).strip() + '>'
