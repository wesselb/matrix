from .matrix import AbstractMatrix

__all__ = ['Diagonal']


class Diagonal(AbstractMatrix):
    """Diagonal matrix.

    Args:
        diag (vector): Diagonal of matrix.
    """

    def __init__(self, diag):
        self.diag = diag


