import lab as B
from lab.shape import Shape

from .matrix import AbstractMatrix, repr_format
from .util import indent, dtype_str
from .shape import assert_matrix

__all__ = ["LowerTriangular", "UpperTriangular"]


class LowerTriangular(AbstractMatrix):
    """Lower-triangular matrix.

    Attributes:
        mat (matrix): Dense lower-triangular matrix.

    Args:
        mat (matrix): Dense lower-triangular matrix.
    """

    def __init__(self, mat):
        assert_matrix(
            mat,
            "Input is not a tensor of at least rank 2. "
            "Can only construct triangular matrices from tensors of at least rank 2.",
        )
        self.mat = B.dense(mat)

    def __str__(self):
        return (
            f"<lower-triangular matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
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
        assert_matrix(
            mat,
            "Input is not a tensor of at least rank 2. "
            "Can only construct triangular matrices from tensors of at least rank 2.",
        )
        self.mat = B.dense(mat)

    def __str__(self):
        return (
            f"<upper-triangular matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
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
