import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..shape import assert_square
from ..triangular import LowerTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _assert_square_cholesky(a):
    assert_square(a, "Can only take the Cholesky decomposition of square matrices.")


@B.dispatch(Zero)
def cholesky(a):
    _assert_square_cholesky(a)
    return a


@B.dispatch(Dense)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = LowerTriangular(B.cholesky(B.reg(a.mat)))
    return a.cholesky


@B.dispatch(Diagonal)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = Diagonal(B.sqrt(a.diag))
    return a.cholesky


@B.dispatch(Constant)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        chol_const = B.divide(B.sqrt(a.const), B.sqrt(a.cols))
        a.cholesky = Constant(chol_const, a.rows, a.cols)
    return a.cholesky


@B.dispatch(LowRank)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        if a.sign == 1:
            a.cholesky = B.matmul(a.left, B.cholesky(a.middle))
        else:
            warn_upmodule(
                f"Cannot ensure positivity of {a}: "
                f"converting to dense to compute the Cholesky "
                f"decomposition.",
                category=ToDenseWarning,
            )
            a.cholesky = B.cholesky(Dense(a))

    return a.cholesky


@B.dispatch(Woodbury)
def cholesky(a):
    if a.cholesky is None:
        warn_upmodule(
            f"Converting {a} to dense to compute its Cholesky decomposition.",
            category=ToDenseWarning,
        )
        a.cholesky = LowerTriangular(B.cholesky(B.reg(B.dense(a))))
    return a.cholesky


@B.dispatch(Kronecker)
def cholesky(a):
    if a.cholesky is None:
        a.cholesky = Kronecker(B.cholesky(a.left), B.cholesky(a.right))
    return a.cholesky
