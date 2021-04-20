import lab as B
from plum import convert

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, AbstractMatrix
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(B.Numeric)
def shape_matrix(a):
    return B.shape_matrix(convert(a, AbstractMatrix))


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def shape_matrix(a):
    return B.shape(a.mat)[-2:]


@B.dispatch(Diagonal)
def shape_matrix(a):
    diag_len = B.shape(a.diag)[-1]
    return diag_len, diag_len


@B.dispatch({Zero, Constant})
def shape_matrix(a):
    return a.rows, a.cols


@B.dispatch(LowRank)
def shape_matrix(a):
    return B.shape_matrix(a.left)[0], B.shape_matrix(a.right)[0]


@B.dispatch(Woodbury)
def shape_matrix(a):
    return B.shape_matrix(a.lr)


@B.dispatch(Kronecker)
def shape_matrix(a):
    left_rows, left_cols = B.shape_matrix(a.left)
    right_rows, right_cols = B.shape_matrix(a.right)
    return left_rows * right_rows, left_cols * right_cols


B.shape_matrix = shape_matrix
