import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Woodbury)
def schur(a):
    if a.schur is None:
        a.schur = B.add(B.inv(a.lr.middle),
                        B.iqf(a.diag, a.lr.right, a.lr.left))
    return a.schur


B.schur = schur
