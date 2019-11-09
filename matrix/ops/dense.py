import lab as B
from ..diagonal import Diagonal

__all__ = []


@B.dispatch(B.Numeric)
def dense(a):
    return a


B.dense = dense


@B.dispatch(Diagonal)
def dense(a):
    return B.diag(a.diag)
