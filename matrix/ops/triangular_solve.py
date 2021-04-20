import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


@B.dispatch
def triangular_solve(a: LowerTriangular, b: AbstractMatrix, lower_a=True):
    if not lower_a:
        warn_upmodule(
            f'Solving against {a}, but "lower_a" is set to "False": ignoring flag.',
            category=UserWarning,
        )
    return B.solve(a, b)


@B.dispatch
def triangular_solve(a: UpperTriangular, b: AbstractMatrix, lower_a=True):
    if lower_a:
        warn_upmodule(
            f'Solving against {a}, but "lower_a" is set to "True": ignoring flag.',
            category=UserWarning,
        )
    return B.solve(a, b)