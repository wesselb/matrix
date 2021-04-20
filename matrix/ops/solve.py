import lab as B
from algebra import proven
from wbml.warning import warn_upmodule

from ..constant import Zero
from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(precedence=proven())
def solve(a: AbstractMatrix, b: Zero):
    return b


@B.dispatch
def solve(a: AbstractMatrix, b: AbstractMatrix):
    if structured(a, b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return B.solve(B.dense(a), B.dense(b))


@B.dispatch
def solve(a: Diagonal, b: AbstractMatrix):
    return B.matmul(B.inv(a), b)


@B.dispatch
def solve(a: LowerTriangular, b: AbstractMatrix):
    if structured(b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=True))


@B.dispatch
def solve(a: UpperTriangular, b: AbstractMatrix):
    if structured(b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=False))


@B.dispatch
def solve(a: Woodbury, b: AbstractMatrix):
    # `B.inv` is optimised with the matrix inversion lemma.
    return B.matmul(B.inv(a), b)