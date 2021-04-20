import abc

import lab as B
import wbml.out
from plum import Dispatcher
from wbml.warning import warn_upmodule

from .shape import assert_matrix
from .util import indent, dtype_str, ToDenseWarning

__all__ = ["AbstractMatrix", "Dense", "repr_format", "structured"]

_dispatch = Dispatcher()


class AbstractMatrix(metaclass=abc.ABCMeta):
    """Abstract matrix type."""

    def __neg__(self):
        return B.negative(self)

    def __add__(self, other):
        return B.add(self, other)

    def __radd__(self, other):
        return B.add(other, self)

    def __sub__(self, other):
        return B.subtract(self, other)

    def __rsub__(self, other):
        return B.subtract(other, self)

    def __mul__(self, other):
        return B.multiply(self, other)

    def __rmul__(self, other):
        return B.multiply(other, self)

    def __truediv__(self, other):
        return B.divide(self, other)

    def __rtruediv__(self, other):
        return B.divide(other, self)

    def __pow__(self, power, modulo=None):
        # TODO: Implement this.
        assert modulo is None, "Modulo in powers is not yet supported."
        return B.power(self, power)

    def __matmul__(self, other):
        return B.matmul(self, other)

    def __getitem__(self, item):
        if structured(self):
            warn_upmodule(
                f"Indexing into {self}: converting to dense.", category=ToDenseWarning
            )
        return B.dense(self)[item]

    @property
    def T(self):
        return B.transpose(self)

    @property
    def shape(self):
        return B.shape(self)

    @property
    def dtype(self):
        return B.dtype(self)

    @abc.abstractmethod
    def __str__(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def __repr__(self):  # pragma: no cover
        pass


class Dense(AbstractMatrix):
    """Dense matrix.

    Attributes:
        mat (matrix): Matrix.
        cholesky (:class:`.triangular.LowerTriangular` or None): Cholesky
            decomposition of the matrix, once it has been computed.

    Args:
        mat (matrix): Matrix.
    """

    def __init__(self, mat):
        assert_matrix(
            mat,
            "Input is not a rank-2 tensor. Can only construct "
            "dense matrices from rank-2 tensors.",
        )
        self.mat = B.dense(mat)
        self.cholesky = None

    def __str__(self):
        rows, cols = B.shape(self)
        return f"<dense matrix: shape={rows}x{cols}, dtype={dtype_str(self)}>"

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" mat="
            + indent(repr_format(self.mat), " " * 5).strip()
            + ">"
        )


@_dispatch
def repr_format(a):
    """Format an object for display in `__repr__` methods.

    Args:
        a (object): Object to format.

    Returns:
        str: String representation of `a`.
    """
    return wbml.out.format(a, info=False)


@_dispatch
def repr_format(a: AbstractMatrix):
    return repr(a)


@_dispatch
def structured(*xs):
    """Check whether there is any structured matrix.

    Args:
        xs (matrices): Matrices to check.

    Returns:
        bool: `True` if there is any structure.
    """
    return any(structured(x) for x in xs)


@_dispatch
def structured(x: AbstractMatrix):
    return True


@_dispatch
def structured(x: Dense):
    return False


@_dispatch
def structured(x: B.Numeric):
    return False