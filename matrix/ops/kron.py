import lab as B
from algebra import proven

from ..constant import Constant, Zero
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix

__all__ = []


def _product_shape(a, b):
    ar, ac = B.shape_matrix(a)
    br, bc = B.shape_matrix(b)
    return ar * br, ac * bc


# Zero


@B.dispatch(precedence=proven())
def kron(a: AbstractMatrix, b: Zero):
    return Zero(B.dtype(b), *B.shape_batch_broadcast(a, b), *_product_shape(a, b))


@B.dispatch(precedence=proven())
def kron(a: Zero, b: AbstractMatrix):
    return kron(b, a)


# Dense


@B.dispatch
def kron(a: AbstractMatrix, b: AbstractMatrix):
    return Kronecker(a, b)


# Constant


@B.dispatch
def kron(a: Constant, b: Constant):
    return Constant(B.multiply(a.const, b.const), *_product_shape(a, b))
