import lab as B
import wbml.out
import abc

from .util import indent

__all__ = ['AbstractMatrix', 'Dense']


class AbstractMatrix:
    """Abstract matrix type."""

    def __neg__(self):
        return B.negative(self)

    def __add__(self, other):
        return B.add(self, other)

    def __radd__(self, other):
        return B.add(other, self)

    def __sub__(self, other):
        return B.subtract(self, other)

    def __rsub__(self, other):
        return B.subtract(other, self)

    def __mul__(self, other):
        return B.multiply(self, other)

    def __rmul__(self, other):
        return B.multiply(other, self)

    def __truediv__(self, other):
        return B.divide(self, other)

    def __rtruediv__(self, other):
        return B.divide(other, self)

    def __pow__(self, power, modulo=None):
        assert modulo is None  # TODO: Implement this.
        return B.power(self, power)

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def __str__(self):  # pragma: no cover
        pass


class Dense(AbstractMatrix):
    """Dense matrix."""

    def __init__(self, mat):
        self.mat = mat

    def __str__(self):
        return (f'Dense matrix:\n'
                f'{indent(wbml.out.format(self.mat))}')
