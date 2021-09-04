import lab as B
from lab.shape import Shape

from .matrix import AbstractMatrix, repr_format
from .shape import assert_matrix
from .util import dtype_str, indent

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
        assert_matrix(
            self.left, f"Left factor is not a tensor of at least rank 2. {msg}"
        )

        # Check middle factor, if it is given.
        if self._middle is not None:
            assert_matrix(
                self._middle, f"Middle factor is not a tensor of at least rank 2. {msg}"
            )
            if B.shape_matrix(self.left, 1) != B.shape_matrix(self._middle, 0):
                raise AssertionError(
                    f"Left factor has {B.shape_matrix(self.left, 1)} column(s), "
                    f"but middle factor has {B.shape_matrix(self.middle, 0)} row(s)."
                )

        # Check right factor, if it is given.
        if self._right is not None:
            assert_matrix(
                self._right, f"Right factor is tensor of at least rank 2. {msg}"
            )

            if self._middle is not None:
                if B.shape_matrix(self._right, 1) != B.shape_matrix(self._middle, 1):
                    raise AssertionError(
                        f"Right factor has {B.shape_matrix(self._right, 1)} column(s), "
                        f"but middle factor has {B.shape_matrix(self.middle, 1)} "
                        f"column(s)."
                    )
            else:
                if B.shape_matrix(self._right, 1) != B.shape_matrix(self.left, 1):
                    raise AssertionError(
                        f"Right factor has {B.shape_matrix(self._right, 1)} column(s), "
                        f"but left factor has {B.shape_matrix(self.left, 1)} column(s)."
                    )

        # Caching attributes:
        self.cholesky = None
        self.dense = None

        # Determine `rank`.
        if self._middle is None:
            self.rank = B.shape_matrix(self.left, 1)
        else:
            self.rank = min(B.shape_matrix(self._middle))

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
        return (
            f"<low-rank matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" dtype={dtype_str(self)},"
            f" rank={self.rank}>"
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
