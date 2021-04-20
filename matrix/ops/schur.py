import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def schur(a: Woodbury):
    """Compute the Schur complement associated to a matrix. A Schur complement will need
    to make sense for the type of `a`.

    Args:
        a (matrix): Matrix to compute Schur complement of.

    Returns:
        matrix: Schur complement.
    """
    if a.schur is None:
        second = B.mm(a.lr.right, B.inv(a.diag), a.lr.left, tr_a=True)
        a.schur = B.add(B.inv(a.lr.middle), second)
    return a.schur


B.schur = schur