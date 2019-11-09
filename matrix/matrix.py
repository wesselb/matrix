import lab as B
from plum import add_promotion_rule, add_conversion_method

__all__ = ['AbstractMatrix']


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

    def __div__(self, other):
        return B.divide(self, other)

    def __rdiv__(self, other):
        return B.divide(other, self)

    def __pow__(self, power, modulo=None):
        assert modulo is None  # TODO: Implement this.
        return B.power(self, power)


add_promotion_rule(AbstractMatrix, B.Numeric, B.Numeric)
add_conversion_method(AbstractMatrix, B.Numeric, lambda x: B.dense(x))
