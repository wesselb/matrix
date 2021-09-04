import lab as B
from lab.shape import Shape

from .matrix import AbstractMatrix, repr_format
from .util import dtype_str, indent
from .shape import expand_and_broadcast

__all__ = ["Zero", "Constant"]


class Zero(AbstractMatrix):
    """Zero matrix.

    Args:
        dtype (dtype): Data type.
        *batch (int, optional): Shape of batch.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Attributes:
        dtype (dtype): Data type.
        batch (tuple): Shape of batch.
        rows (int): Number of rows.
        cols (int): Number of columns.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.
    """

    def __init__(self, dtype, *shape):
        # Check that at least a rank-2 tensor is specified.
        if len(shape) < 2:
            raise ValueError("Must specify the number of rows and columns.")
        self._dtype = dtype
        self.batch = shape[:-2]
        self.rows = shape[-2]
        self.cols = shape[-1]
        self.dense = None

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return (
            f"<zero matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" dtype={dtype_str(self.dtype)}>"
        )

    def __repr__(self):
        return str(self)


class Constant(AbstractMatrix):
    """Constant matrix.

    Attributes:
        const (scalar): Constant of the matrix.
        batch (tuple): Shape of batch.
        rows (int): Number of rows.
        cols (int): Number of columns.
        cholesky (:class:`.constant.Constant` or None): Cholesky
            decomposition of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        const (scalar): Constant.
        *batch (int, optional): Shape of batch.
        rows (int): Number of rows.
        cols (int): Number of columns.
    """

    def __init__(self, const, *shape):
        # Check that at least a rank-2 tensor is specified.
        if len(shape) < 2:
            raise ValueError("Must specify the number of rows and columns.")
        self.rows = shape[-2]
        self.cols = shape[-1]
        if shape[:-2] == ():
            # No batch shape was specified. Infer it from `const`.
            self.batch = B.shape(const)
            self.const = const
        else:
            # A batch shape was specified. Broadcast it with the shape of `const`.
            self.batch = expand_and_broadcast(B.shape(const), shape[:-2])
            self.const = B.broadcast_to(const, *self.batch)
        self.cholesky = None
        self.dense = None

    def __str__(self):
        return (
            f"<constant matrix:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" const="
            + indent(repr_format(self.const), " " * 7).strip()
            + ">"
        )
