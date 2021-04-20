import lab as B

__all__ = []


@B.dispatch
def ratio(a, b):
    """Compute the ratio between two positive-definite matrices.

    Args:
        a (matrix): Numerator.
        b (matrix): Denominator.

    Returns:
        matrix: Ratio.
    """
    chol = B.cholesky(a)
    return B.sum(B.iqf_diag(b, chol))


B.ratio = ratio