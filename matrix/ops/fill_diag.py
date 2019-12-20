import lab as B

from ..diagonal import Diagonal

__all__ = []


@B.dispatch(B.Numeric, B.Int)
def fill_diag(a, diag_len):
    """Fill the diagonal of a diagonal matrix with a particular scalar.

    Args:
        a (scalar): Scalar to fill diagonal with.
        diag_len (int): Length of diagonal.
    """
    return Diagonal(a * B.ones(B.dtype(a), diag_len))


B.fill_diag = fill_diag
