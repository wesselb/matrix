import lab as B

from .matrix import AbstractMatrix, repr_format
from .shape import assert_matrix
from .util import indent, dtype_str

__all__ = ["LowRank"]


class LowRank(AbstractMatrix):
    """Abstract low-rank matrix.

    The data type of a low-rank matrix is the data type of the left factor. The
    Cholesky decomposition is not exactly the Cholesky decomposition, but returns a
    matrix `L` such that `L transpose(L)` is the original matrix.

    Attributes:
        left (matrix): Left factor.
        middle (matrix): Middle factor.
        right (matrix): Right factor.
        rank (int): Rank of the low-rank matrix.
        cholesky (:class:`.matrix.AbstractMatrix` or None): Cholesky-like
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been computed.

    Args:
        left (matrix): Left factor.
        right (matrix, optional): Right factor. Defaults to the left factor.
        middle (matrix, optional): Middle factor. Defaults to the identity matrix.
    """

    def __init__(self, left, right=None, middle=None):
        self.left = left
        self._right = right
        self._middle = middle
        self._middle_default = None

        msg = "Can only construct low-rank matrices from matrix factors."

        # Check left factor.
        assert_matrix(self.left, f"Left factor is not a rank-2 tensor. {msg}")

        # Check middle factor, if it is given.
        if self._middle is not None:
            assert_matrix(self._middle, f"Middle factor is not a rank-2 tensor. {msg}")
            if B.shape(self.left)[1] != B.shape(self._middle)[0]:
                raise AssertionError(
                    f"Left factor has {B.shape(self.left)[1]} columns "
                    f"and middle factor has {B.shape(self.middle)[0]} rows."
                )

        # Check right factor, if it is given.
        if self._right is not None:
            assert_matrix(self._right, f"Right factor is not a rank-2 tensor. {msg}")

            if self._middle is not None:
                if B.shape(self._right)[1] != B.shape(self._middle)[1]:
                    raise AssertionError(
                        f"Right factor has {B.shape(self._right)[1]} columns "
                        f"and middle factor has {B.shape(self.middle)[1]} columns."
                    )
            else:
                if B.shape(self._right)[1] != B.shape(self.left)[1]:
                    raise AssertionError(
                        f"Right factor has {B.shape(self._right)[1]} columns and "
                        f"left factor has {B.shape(self.left)[1]} columns."
                    )

        # Caching attributes:
        self.cholesky = None
        self.dense = None

        # Determine `rank`.
        if self._middle is None:
            self.rank = B.shape(self.left)[1]
        else:
            self.rank = min(B.shape(self._middle))

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
            f" dtype={dtype_str(self)}," + f" rank={self.rank}>"
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
