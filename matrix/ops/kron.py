import lab as B
from algebra import proven

from ..constant import Zero, Constant
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix

__all__ = []


def _product_shape(a, b):
    ar, ac = B.shape(a)
    br, bc = B.shape(b)
    return ar * br, ac * bc


# Zero


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def kron(a, b):
    return Zero(B.dtype(b), *_product_shape(a, b))


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def kron(a, b):
    return kron(b, a)


# Dense


@B.dispatch(AbstractMatrix, AbstractMatrix)
def kron(a, b):
    return Kronecker(a, b)


# Constant


@B.dispatch(Constant, Constant)
def kron(a, b):
    return Constant(B.multiply(a.const, b.const), *_product_shape(a, b))
