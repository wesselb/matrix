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


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def solve(a, b):
    return b


@B.dispatch(AbstractMatrix, AbstractMatrix)
def solve(a, b):
    if structured(a, b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return B.solve(B.dense(a), B.dense(b))


@B.dispatch(Diagonal, AbstractMatrix)
def solve(a, b):
    return B.matmul(B.inv(a), b)


@B.dispatch(LowerTriangular, AbstractMatrix)
def solve(a, b):
    if structured(b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=True))


@B.dispatch(UpperTriangular, AbstractMatrix)
def solve(a, b):
    if structured(b):
        warn_upmodule(
            f"Solving {a} x = {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=False))


@B.dispatch(Woodbury, AbstractMatrix)
def solve(a, b):
    # `B.inv` is optimised with the matrix inversion lemma.
    return B.matmul(B.inv(a), b)
