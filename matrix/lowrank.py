import lab as B

from .matrix import AbstractMatrix, repr_format
from .shape import assert_matrix, assert_square
from .util import indent, dtype_str

__all__ = ['LowRank']


class LowRank(AbstractMatrix):
    """Low-rank matrix.

    The data type of a low-rank matrix is the data type of the left factor.
    The Cholesky decomposition is not exactly the Cholesky decomposition,
    but returns a matrix `L` such that `L transpose(L)` is the original matrix.

    Attributes:
        left (matrix): Left factor.
        middle (matrix): Middle factor.
        right (matrix): Right factor.
        rank (int): Rank of the low-rank matrix.
        symmetric (bool): Boolean indicating whether this the left and right
            factor are identical. Note that this attribute does not check
            whether the middle factor is symmetric.
        cholesky (:class:`.matrix.AbstractMatrix` or None): Cholesky-like
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        left (matrix): Left factor.
        middle (matrix, optional): Middle factor. Defaults to the identity
            matrix.
        right (matrix, optional): Right factor. Defaults to the left factor.
    """

    def __init__(self, left, right=None, middle=None):
        self.left = left
        self.rank = B.shape(self.left)[1]
        self.right = left if right is None else right
        if middle is None:
            self.middle = B.fill_diag(B.one(self.left), self.rank)
        else:
            self.middle = middle
        self.symmetric = right is None
        self.cholesky = None
        self.dense = None

        msg = 'Can only construct low-rank matrices from matrix factors and ' \
              'the middle factor must be square.'
        assert_matrix(self.left,
                      f'Left factor is not a rank-2 tensor. {msg}')
        assert_square(self.middle,
                      f'Middle factor is not a square rank-2 tensor. {msg}')
        assert_matrix(self.right,
                      f'Right factor is not a rank-2 tensor. {msg}')

        if (
                B.shape(self.middle)[1] != self.rank or
                B.shape(self.right)[1] != self.rank
        ):
            raise ValueError('All factors must have an equal number '
                             'of columns.')

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<low-rank matrix:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},' + \
               f' rank={self.rank},' \
               f' symmetric={self.symmetric}>'

    def __repr__(self):
        return str(self)[:-1] + '\n' + \
               f' left=' + indent(repr_format(self.left),
                                  ' ' * 6).strip() + '\n' + \
               f' middle=' + indent(repr_format(self.middle),
                                    ' ' * 8).strip() + '\n' + \
               f' right=' + indent(repr_format(self.right),
                                   ' ' * 7).strip() + '>'
