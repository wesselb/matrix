import lab as B
from plum import add_promotion_rule, add_conversion_method

from .diagonal import Diagonal
from .matrix import AbstractMatrix, Dense

__all__ = []

add_promotion_rule(AbstractMatrix, B.Numeric, AbstractMatrix)
add_conversion_method(B.Numeric, AbstractMatrix, Dense)

add_promotion_rule(Diagonal, Dense, Dense)
add_conversion_method(Diagonal, Dense, lambda x: Dense(B.dense(x)))
