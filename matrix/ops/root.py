import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..shape import assert_square
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _assert_square_root(a):
    assert_square(a, "Can only take a square root of square matrices.")


@B.dispatch(B.Numeric)
def root(a):  # pragma: no cover
    """Compute the positive square root of a positive-definite matrix.

    Args:
        a (matrix): Matrix to compute square root of.

    Returns:
        matrix: Positive square root of `a`.
    """
    _assert_square_root(a)
    u, s, _ = B.svd(a)
    return B.mm(u, B.diag(B.sqrt(s)), u, tr_c=True)


B.root = root


@B.dispatch(Zero)
def root(a):
    _assert_square_root(a)
    return a


@B.dispatch(Dense)
def root(a):
    return Dense(B.root(B.dense(a)))


@B.dispatch(Diagonal)
def root(a):
    _assert_square_root(a)
    return B.cholesky(a)


@B.dispatch(Constant)
def root(a):
    _assert_square_root(a)
    return B.cholesky(a)


@B.dispatch({LowRank, Woodbury})
def root(a):
    warn_upmodule(
        f"Converting {a} to dense to compute its square root.", category=ToDenseWarning
    )
    return Dense(B.root(B.dense(a)))


@B.dispatch(Kronecker)
def root(a):
    return Kronecker(B.root(a.left), B.root(a.right))
