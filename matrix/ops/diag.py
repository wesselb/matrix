import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _diag_len(a):
    return B.minimum(*B.shape(a))


@B.dispatch(Zero)
def diag(a):
    return B.zeros(B.dtype(a), _diag_len(a))


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def diag(a):
    return B.diag(a.mat)


@B.dispatch(Diagonal)
def diag(a):
    return a.diag


@B.dispatch(Constant)
def diag(a):
    return a.const * B.ones(B.dtype(a), _diag_len(a))


@B.dispatch(LowRank)
def diag(a):
    if structured(a.left, a.right):
        warn_upmodule(
            f"Getting the diagonal of {a}: converting the factors to dense.",
            category=ToDenseWarning,
        )
    diag_len = _diag_len(a)
    left_mul = B.matmul(a.left, a.middle)
    return B.sum(
        B.multiply(B.dense(left_mul)[:diag_len, :], B.dense(a.right)[:diag_len, :]),
        axis=1,
    )


@B.dispatch(Woodbury)
def diag(a):
    return B.diag(a.diag) + B.diag(a.lr)


@B.dispatch(Kronecker)
def diag(a):
    return B.kron(B.diag(a.left), B.diag(a.right))


# Construct block-diagonal matrices.


@B.dispatch(object, object)
def diag(a, b):
    # We could merge this with `block`, but `block` has a lot of overhead. It
    # seems advantageous to optimise this common case.
    warn_upmodule(
        f"Constructing a dense block-diagonal matrix from "
        f"{a} and {b}: converting to dense.",
        category=ToDenseWarning,
    )
    a = B.dense(a)
    b = B.dense(b)

    dtype = B.dtype(a)
    ar, ac = B.shape(a)
    br, bc = B.shape(b)
    return Dense(B.concat2d([a, B.zeros(dtype, ar, bc)], [B.zeros(dtype, br, ac), b]))


@B.dispatch(Diagonal, Diagonal)
def diag(a, b):
    return Diagonal(B.concat(a.diag, b.diag))
