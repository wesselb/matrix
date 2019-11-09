from .matrix import AbstractMatrix
import wbml.out
from .util import indent

__all__ = ['Diagonal']


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Args:
        diag (vector): Diagonal of matrix.
    """

    def __init__(self, diag):
        self.diag = diag

    def __str__(self):
        return (f'Diagonal matrix with diagonal\n'
                f'{indent(wbml.out.format(self.diag))}')
