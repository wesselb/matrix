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


@B.dispatch
def cholesky(a: Zero):
    _assert_square_cholesky(a)
    return a


@B.dispatch
def cholesky(a: Dense):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = LowerTriangular(B.cholesky(B.reg(a.mat)))
    return a.cholesky


@B.dispatch
def cholesky(a: Diagonal):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = Diagonal(B.sqrt(a.diag))
    return a.cholesky


@B.dispatch
def cholesky(a: Constant):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        chol_const = B.divide(B.sqrt(a.const), B.sqrt(a.cols))
        a.cholesky = Constant(chol_const, a.rows, a.cols)
    return a.cholesky


@B.dispatch
def cholesky(a: LowRank):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = B.matmul(a.left, B.cholesky(a.middle))
    return a.cholesky


@B.dispatch
def cholesky(a: Woodbury):
    if a.cholesky is None:
        warn_upmodule(
            f"Converting {a} to dense to compute its Cholesky decomposition.",
            category=ToDenseWarning,
        )
        a.cholesky = LowerTriangular(B.cholesky(B.reg(B.dense(a))))
    return a.cholesky


@B.dispatch
def cholesky(a: Kronecker):
    if a.cholesky is None:
        a.cholesky = Kronecker(B.cholesky(a.left), B.cholesky(a.right))
    return a.cholesky