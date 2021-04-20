import lab as B
from lab.shape import Shape
from plum import Dispatcher, Self

from .matrix import AbstractMatrix, repr_format
from .shape import assert_vector
from .util import indent, dtype_str

__all__ = ["Diagonal"]


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Attributes:
        diag (vector): Diagonal of the matrix.
        cholesky (:class:`.constant.Diagonal` or none): Cholesky
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        diag (vector): Diagonal of matrix.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric)
    def __init__(self, diag):
        assert_vector(
            diag,
            "Input is not a tensor of at least rank 1. "
            "Can only construct diagonal matrices from tensor of at least rank 1.",
        )
        self.diag = diag
        self.cholesky = None
        self.dense = None

    @_dispatch(AbstractMatrix)
    def __init__(self, mat):
        return Diagonal.__init__(self, B.diag_extract(mat))

    def __str__(self):
        return (
            f"<diagonal matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" diag="
            + indent(repr_format(self.diag), " " * 6).strip()
            + ">"
        )
