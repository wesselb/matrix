import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def shape(a):
    return B.shape(a.mat)


@B.dispatch(Diagonal)
def shape(a):
    diag_len = B.shape(a.diag)[0]
    return diag_len, diag_len


@B.dispatch({Zero, Constant})
def shape(a):
    return a.rows, a.cols


@B.dispatch(LowRank)
def shape(a):
    return B.shape(a.left)[0], B.shape(a.right)[0]


@B.dispatch(Woodbury)
def shape(a):
    return B.shape(a.lr)


@B.dispatch(Kronecker)
def shape(a):
    left_rows, left_cols = B.shape(a.left)
    right_rows, right_cols = B.shape(a.right)
    return left_rows * right_rows, left_cols * right_cols
