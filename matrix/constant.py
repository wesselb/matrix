import lab as B

from .matrix import AbstractMatrix, repr_format
from .shape import assert_scalar
from .util import dtype_str

__all__ = ["Zero", "Constant"]


class Zero(AbstractMatrix):
    """Zero matrix.

    Attributes:
        dtype (dtype): Data type.
        rows (int): Number of rows.
        cols (int): Number of columns.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        dtype (dtype): Data type.
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, dtype, rows, cols):
        self._dtype = dtype
        self.rows = rows
        self.cols = cols
        self.dense = None

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        rows, cols = B.shape(self)
        return f"<zero matrix: shape={rows}x{cols}, dtype={dtype_str(self.dtype)}>"

    def __repr__(self):
        return str(self)


class Constant(AbstractMatrix):
    """Constant matrix.

    Attributes:
        const (scalar): Constant of the matrix.
        rows (int): Number of rows.
        cols (int): Number of columns.
        cholesky (:class:`.constant.Constant` or None): Cholesky
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        const (scalar): Constant.
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, const, rows, cols):
        assert_scalar(
            const,
            "Input is not a scalar. Can only construct constant matrices from scalars.",
        )
        self.const = const
        self.rows = rows
        self.cols = cols
        self.cholesky = None
        self.dense = None

    def __str__(self):
        rows, cols = B.shape(self)
        return (
            f"<constant matrix:"
            f" shape={rows}x{cols},"
            f" dtype={dtype_str(self)},"
            f" const={repr_format(self.const)}>"
        )

    def __repr__(self):
        return str(self)
