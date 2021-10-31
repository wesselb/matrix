from typing import Union

import lab as B
from wbml.warning import warn_upmodule

from .util import align_batch
from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _diag_len(a):
    return B.minimum(*B.shape_matrix(a))


@B.dispatch
def diag(a: Zero):
    return B.zeros(B.dtype(a), *B.shape_batch(a), _diag_len(a))


@B.dispatch
def diag(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.diag_extract(a.mat)


@B.dispatch
def diag(a: Diagonal):
    return a.diag


@B.dispatch
def diag(a: Constant):
    ones = B.ones(B.dtype(a), *B.shape_batch(a), _diag_len(a))
    return B.expand_dims(a.const, axis=-1, ignore_scalar=True) * ones


@B.dispatch
def diag(a: LowRank):
    if structured(a.left, a.right):
        warn_upmodule(
            f"Getting the diagonal of {a}: converting the factors to dense.",
            category=ToDenseWarning,
        )
    diag_len = _diag_len(a)
    return B.sum(
        B.multiply(
            B.dense(B.matmul(a.left, a.middle))[..., :diag_len, :],
            B.dense(a.right)[..., :diag_len, :],
        ),
        axis=-1,
    )


@B.dispatch
def diag(a: Woodbury):
    return B.diag(a.diag) + B.diag(a.lr)


@B.dispatch
def diag(a: Kronecker):
    left, right = align_batch(a.left, a.right)
    return B.kron(B.diag(left), B.diag(right), -1)
