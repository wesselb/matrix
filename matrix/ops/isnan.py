import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, structured
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def isnan(a: AbstractMatrix):
    if structured(a):
        warn_upmodule(
            f'Applying "isnan" to {a}: converting to dense.', category=ToDenseWarning
        )
    return B.isnan(B.dense(a))