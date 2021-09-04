import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def sample(state: B.RandomState, a, num: B.Int = 1):
    """Sample from covariance matrices.

    Args:
        state (random state, optional): Random state.
        a (tensor): Covariance matrix to sample from.
        num (int): Number of samples.

    Returns:
        tensor: Samples as rank 2 column vectors.
    """
    chol = B.cholesky(a)
    state, noise = B.randn(
        state,
        B.dtype_float(a),
        *B.shape_batch(a),
        B.shape(chol, -1),
        num,
    )
    return state, B.matmul(chol, noise)


B.sample = sample


@B.dispatch
def sample(a, num: B.Int = 1):
    state, res = sample(B.global_random_state(a), a, num=num)
    B.set_global_random_state(state)
    return res


@B.dispatch
def sample(state: B.RandomState, a: Woodbury, num: B.Int = 1):
    state, sample_diag = B.sample(state, a.diag, num=num)
    state, sample_lr = B.sample(state, a.lr, num=num)
    return state, sample_diag + sample_lr
