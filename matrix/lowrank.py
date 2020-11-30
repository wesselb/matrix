import lab as B

from .matrix import AbstractMatrix, repr_format
from .shape import assert_matrix, assert_square
from .util import indent, dtype_str

__all__ = ["LowRank"]


class LowRank(AbstractMatrix):
    """Abstract low-rank matrix.

    The data type of a low-rank matrix is the data type of the left factor.
    The Cholesky decomposition is not exactly the Cholesky decomposition,
    but returns a matrix `L` such that `L transpose(L)` is the original matrix.

    Attributes:
        left (matrix): Left factor.
        middle (matrix): Middle factor.
        right (matrix): Right factor.
        rank (int): Rank of the low-rank matrix.
        sign (int): Definiteness of the matrix, which is `1` if the
            matrix is PD, `0` if it is indefinite, and `-1` if it is ND.
        cholesky (:class:`.matrix.AbstractMatrix` or None): Cholesky-like
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        left (matrix): Left factor.
        right (matrix, optional): Right factor. Defaults to the left factor.
        middle (matrix, optional): Middle factor. Defaults to the identity
            matrix.
        sign (int, optional): Definiteness of the matrix. Defaults to `1` if
            `left` is identical to `right` and `0` otherwise.
    """

    def __init__(self, left, right=None, middle=None, sign=None):
        self.left = left
        self._right = right
        self._middle = middle
        self._middle_default = None

        # Set `sign` and `rank`.
        self.rank = B.shape(self.left)[1]
        if sign is None:
            if self.left is self.right:
                self.sign = 1
            else:
                self.sign = 0
        else:
            self.sign = sign

        # Caching attributes:
        self.cholesky = None
        self.dense = None

        msg = (
            "Can only construct low-rank matrices from matrix factors and "
            "the middle factor must be square."
        )

        # Check left factor.
        assert_matrix(self.left, f"Left factor is not a rank-2 tensor. {msg}")

        # Check right factor, if it is given.
        if self._right is not None:
            assert_matrix(self._right, f"Right factor is not a rank-2 tensor. {msg}")
            if B.shape(self._right)[1] != self.rank:
                raise AssertionError(
                    "All factors must have an equal number of columns."
                )

        # Check middle factor, if it is given.
        if self._middle is not None:
            assert_square(
                self._middle, f"Middle factor is not a square rank-2 tensor. {msg}"
            )
            if B.shape(self._middle)[1] != self.rank:
                raise AssertionError(
                    "All factors must have an equal number of columns."
                )

    @property
    def right(self):
        """Right factor."""
        if self._right is None:
            return self.left
        else:
            return self._right

    @property
    def middle(self):
        """Middle factor."""
        if self._middle is None:
            if self._middle_default is None:
                self._middle_default = B.fill_diag(B.one(self.left), self.rank)
            return self._middle_default
        else:
            return self._middle

    def __str__(self):
        rows, cols = B.shape(self)
        return (
            f"<low-rank matrix:"
            f" shape={rows}x{cols},"
            f" dtype={dtype_str(self)}," + f" rank={self.rank},"
            f" sign={self.sign}>"
        )

    def __repr__(self):
        out = (
            str(self)[:-1]
            + "\n left="
            + indent(repr_format(self.left), " " * 6).strip()
        )

        if self._middle is not None:
            out += "\n middle=" + indent(repr_format(self.middle), " " * 8).strip()

        if self._right is not None:
            out += "\n right=" + indent(repr_format(self.right), " " * 7).strip()

        return out + ">"
