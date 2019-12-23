import warnings

import lab as B

from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix, AbstractMatrix)
def divide(a, b):
    if structured(a, b):
        warnings.warn(f'Dividing {a} by {b}: converting to dense.',
                      category=ToDenseWarning)
    return Dense(B.divide(B.dense(a), B.dense(b)))
