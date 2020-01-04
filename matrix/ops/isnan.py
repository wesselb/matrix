import lab as B
import warnings

from ..matrix import AbstractMatrix, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(AbstractMatrix)
def isnan(a):
    if structured(a):
        warnings.warn(f'Applying "isnan" to {a}: converting to dense.',
                      category=ToDenseWarning)
    return B.isnan(B.dense(a))
