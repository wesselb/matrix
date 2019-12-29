import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Woodbury)
def schur(a):
    if a.schur is None:
        second = B.mm(a.lr.right, B.inv(a.diag), a.lr.left, tr_a=True)
        a.schur = B.add(B.inv(a.lr.middle), second)
    return a.schur


B.schur = schur
