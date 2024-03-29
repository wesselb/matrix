import lab as B

from ..diagonal import Diagonal

__all__ = []


@B.dispatch
def fill_diag(a: B.Numeric, diag_len: B.Int):
    """Fill the diagonal of a diagonal matrix with a particular scalar.

    Args:
        a (scalar): Scalar to fill diagonal with.
        diag_len (int): Length of diagonal.
    """
    return Diagonal(
        B.expand_dims(a, axis=-1) * B.ones(B.dtype(a), *B.shape(a), diag_len)
    )


B.fill_diag = fill_diag
