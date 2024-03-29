import lab as B
from lab.shape import Shape
from plum import Dispatcher

from .diagonal import Diagonal
from .lowrank import LowRank
from .matrix import AbstractMatrix, repr_format
from .shape import assert_compatible
from .util import dtype_str, indent

__all__ = ["Woodbury"]

_dispatch = Dispatcher()


class Woodbury(AbstractMatrix):
    """Woodbury matrix.

    The shape and data type of a Woodbury matrix are the shape and data type of
    the low-rank part.

    Attributes:
        diag (:class:`.diagonal.Diagonal`): Diagonal part.
        lr (:class:`.diagonal.LowRank`): Low-rank part.
        cholesky (:class:`.triangular.LowerTriangular` or None): Cholesky
            decomposition of the matrix, once it has been computed.
        schur (:class:`.matrix.AbstractMatrix` or None): Schur complement
            of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        diag (:class:`.diagonal.Diagonal`): Diagonal part.
        lr (:class:`.diagonal.LowRank`): Low-rank part.
    """

    @_dispatch
    def __init__(self, diag: Diagonal, lr: LowRank):
        assert_compatible(B.shape(diag), B.shape(lr))
        self.diag = diag
        self.lr = lr
        self.cholesky = None
        self.schur = None
        self.dense = None

    def __str__(self):
        return (
            f"<Woodbury matrix:"
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
            + "\n"
            + f" lr="
            + indent(repr_format(self.lr), " " * 4).strip()
            + ">"
        )
