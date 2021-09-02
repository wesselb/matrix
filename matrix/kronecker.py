import lab as B
from lab.shape import Shape

from .matrix import AbstractMatrix, repr_format
from .util import dtype_str, indent

__all__ = ["Kronecker"]


class Kronecker(AbstractMatrix):
    """Kronecker product.

    The data type of a Kronecker product is the data type of the left matrix
    in the product.

    Attributes:
        left (matrix): Left matrix in the product.
        right (matrix): Right matrix in the product.
        cholesky (:class:`.kronecker.Kronecker` or None): Cholesky decomposition
            of the matrix, once it has been computed.
        dense (matrix or None): Dense version of the matrix, once it has been
            computed.

    Args:
        left (matrix): Left matrix in the product.
        right (matrix): Right matrix in the product.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.cholesky = None
        self.dense = None

    def __str__(self):
        return (
            f"<Kronecker product:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        return (
            str(self)[:-1]
            + "\n"
            + f" left="
            + indent(repr_format(self.left), " " * 6).strip()
            + "\n"
            + f" right="
            + indent(repr_format(self.right), " " * 7).strip()
            + ">"
        )
