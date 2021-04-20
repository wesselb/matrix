import lab as B

from .matrix import AbstractMatrix, repr_format
from .util import indent, dtype_str

__all__ = ["LowerTriangular", "UpperTriangular"]


class LowerTriangular(AbstractMatrix):
    """Lower-triangular matrix.

    Attributes:
        mat (matrix): Dense lower-triangular matrix.

    Args:
        mat (matrix): Dense lower-triangular matrix.
    """

    def __init__(self, mat):
        self.mat = B.dense(mat)

    def __str__(self):
        rows, cols = B.shape(self)
        return (
            f"<lower-triangular matrix:"
            f" shape={rows}x{cols},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" mat="
            + indent(repr_format(self.mat), " " * 5).strip()
            + ">"
        )


class UpperTriangular(AbstractMatrix):
    """Upper-triangular matrix.

    Attributes:
        mat (matrix): Dense upper-triangular matrix.

    Args:
        mat (matrix): Dense upper-triangular matrix.
    """

    def __init__(self, mat):
        self.mat = B.dense(mat)

    def __str__(self):
        rows, cols = B.shape(self)
        return (
            f"<upper-triangular matrix:"
            f" shape={rows}x{cols},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" mat="
            + indent(repr_format(self.mat), " " * 5).strip()
            + ">"
        )
