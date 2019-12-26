import lab as B
import warnings
from algebra import proven

from ..matrix import AbstractMatrix, Dense, structured
from ..woodbury import Woodbury
from ..triangular import LowerTriangular, UpperTriangular
from ..diagonal import Diagonal
from ..constant import Zero
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def solve(a, b):
    return b


@B.dispatch(AbstractMatrix, AbstractMatrix)
def solve(a, b):
    if structured(a, b):
        warnings.warn(f'Solving {a} x = {b}: converting to dense.',
                      category=ToDenseWarning)
    return B.solve(B.dense(a), B.dense(b))


@B.dispatch(Diagonal, AbstractMatrix)
def solve(a, b):
    return B.matmul(B.inv(a), b)


@B.dispatch(LowerTriangular, AbstractMatrix)
def solve(a, b):
    if structured(b):
        warnings.warn(f'Solving {a} x = {b}: converting to dense.',
                      category=ToDenseWarning)
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=True))


@B.dispatch(UpperTriangular, AbstractMatrix)
def solve(a, b):
    if structured(b):
        warnings.warn(f'Solving {a} x = {b}: converting to dense.',
                      category=ToDenseWarning)
    return Dense(B.trisolve(a.mat, B.dense(b), lower_a=False))


@B.dispatch(Woodbury, AbstractMatrix)
def solve(a, b):
    # Use the matrix inversion lemma:
    inv_diag = B.inv(a.diag)
    left = B.mm(inv_diag, a.lr.left)
    right = B.mm(inv_diag, a.lr.right)
    second = B.mm(left, B.solve(B.dense(B.schur(a)), B.mm(right, b, tr_a=True)))
    return B.subtract(B.mm(inv_diag, b), second)
