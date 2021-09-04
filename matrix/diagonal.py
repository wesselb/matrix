import lab as B
from lab.shape import Shape
from plum import Dispatcher

from .matrix import AbstractMatrix, repr_format
from .shape import assert_vector
from .util import dtype_str, indent

__all__ = ["Diagonal"]

_dispatch = Dispatcher()


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Attributes:
        diag (vector): Diagonal of the matrix.
        cholesky (:class:`.constant.Diagonal` or None): Cholesky decomposition of the
            matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been computed.

    Args:
        diag (vector): Diagonal of matrix.
    """

    @_dispatch
    def __init__(self, diag: B.Numeric):
        assert_vector(
            diag,
            "Input is not a tensor of at least rank 1. "
            "Can only construct diagonal matrices from tensor of at least rank 1.",
        )
        self.diag = diag
        self.cholesky = None
        self.dense = None

    @_dispatch
    def __init__(self, mat: AbstractMatrix):
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
