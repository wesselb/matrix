import abc
from typing import Union

import lab as B
import wbml.out
from lab.shape import Shape
from plum import Dispatcher
from wbml.warning import warn_upmodule

from .util import ToDenseWarning, dtype_str, indent

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

    def __rmatmul__(self, other):
        return B.matmul(other, self)

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

    def squeeze(self):
        return B.squeeze(self)

    @_dispatch
    def reshape(self, shape: Union[tuple, list]):
        return B.reshape(self, *shape)

    @_dispatch
    def reshape(self, *shape: B.Int):
        return B.reshape(self, *shape)

    def flatten(self):
        return self.reshape(-1)

    def diagonal(self):
        return B.diag(self)

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
        self.mat = B.uprank(B.dense(mat))
        self.cholesky = None

    def __str__(self):
        return (
            f"<dense matrix:"
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


@_dispatch
def repr_format(a):
    """Format an object for display in `__repr__` methods.

    Args:
        a (object): Object to format.

    Returns:
        str: String representation of `a`.
    """
    return wbml.out.format(a, False)


@_dispatch
def repr_format(a: AbstractMatrix):
    return repr(a)


@_dispatch
def structured(*xs):
    """Check whether there is any structured matrix.

    Args:
        *xs (matrix): Matrices to check.

    Returns:
        bool: `True` if there is any structure.
    """
    return any(structured(x) for x in xs)


@_dispatch
def structured(x):
    raise RuntimeError(
        f"Could not determine whether {x} is a structured matrix or not."
    )


@_dispatch
def structured(x: AbstractMatrix):
    return True


@_dispatch
def structured(x: Dense):
    return False


@_dispatch
def structured(x: B.Numeric):
    return False
