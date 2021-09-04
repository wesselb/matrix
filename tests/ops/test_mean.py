import lab as B
import pytest

# noinspection PyUnresolvedReferences
from ..util import ConditionalContext, AssertDenseWarning, approx, check_un_op, dense_r


def _check_mean(a, warn_batch=False):
    for axis in [None, B.rank(a) - 1, B.rank(a) - 2, -1, -2]:
        check_un_op(lambda x: B.mean(x, axis=axis), a)

    # Also test summing over the batch dimension!
    if B.shape_batch(a) != ():
        for axis in [0, -3]:
            with ConditionalContext(
                warn_batch, AssertDenseWarning("over batch dimensions")
            ):
                check_un_op(lambda x: B.mean(x, axis=axis), a)

    with pytest.raises(ValueError):
        B.mean(a, axis=5)


def test_sum_dense(dense_r):
    _check_mean(dense_r)
