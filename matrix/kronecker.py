import lab as B
from plum import Dispatcher, Self

from .matrix import AbstractMatrix
from .util import indent, dtype_str

__all__ = ['Kronecker']


class Kronecker(AbstractMatrix):
    """Kronecker product.

    The data type of a Kronecker product is the data type of the left matrix
    in the product.

    Attributes:
        left (:class:`.matrix.AbstractMatrix`): Left matrix in the product.
        right (:class:`.matrix.AbstractMatrix`): Right matrix in the product.

    Args:
        left (:class:`.matrix.AbstractMatrix`): Left matrix in the product.
        right (:class:`.matrix.AbstractMatrix`): Right matrix in the product.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(AbstractMatrix, AbstractMatrix)
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<Kronecker product:' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' left=' + indent(str(self.left), ' ' * 6).strip() + ',\n' + \
               f' right=' + indent(str(self.right), ' ' * 7).strip() + '>'