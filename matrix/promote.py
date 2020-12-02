import lab as B
from plum import add_promotion_rule, conversion_method

from .constant import Zero, Constant
from .lowrank import LowRank
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
        raise RuntimeError(f"Cannot convert rank {B.rank(x)} input to a matrix.")


@conversion_method(Constant, LowRank)
def constant_to_lowrank(a):
    dtype = B.dtype(a)
    rows, cols = B.shape(a)
    middle = B.fill_diag(a.const, 1)
    if rows == cols:
        return LowRank(B.ones(dtype, rows, 1), middle=middle)
    else:
        return LowRank(
            B.ones(dtype, rows, 1), B.ones(dtype, cols, 1), middle=middle
        )
