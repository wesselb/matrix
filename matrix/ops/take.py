import lab as B
import warnings

from ..matrix import AbstractMatrix, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix, object)
def take(a, indices_or_mask, axis=0):
    if structured(a):
        warnings.warn(f'Taking from {a}: converting to dense.',
                      category=ToDenseWarning)
    return B.take(B.dense(a), indices_or_mask, axis=axis)
