import lab as B
import warnings
from plum import add_promotion_rule, conversion_method

from .constant import Zero, Constant
from .matrix import AbstractMatrix, Dense

__all__ = []

add_promotion_rule(AbstractMatrix, B.Numeric, AbstractMatrix)
add_promotion_rule(AbstractMatrix, AbstractMatrix, AbstractMatrix)


@conversion_method(B.Numeric, AbstractMatrix)
def convert(x):
    if B.rank(x) == 0:
        if isinstance(x, B.Number) and x == 0:
            return Zero(B.dtype(x), 1, 1)
        else:
            return Constant(x, 1, 1)
    elif B.rank(x) == 2:
        return Dense(x)
    else:
        raise RuntimeError(f'Cannot convert rank {B.rank(x)} input '
                           f'to a matrix.')
