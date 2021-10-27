import pytest
import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1


def test_uprank_dense(dense1):
    assert B.uprank(dense1) is dense1

    # Test that `rank` can be given as a keyword argument, but must also be set to `2`.
    B.uprank(dense1, rank=2)
    with pytest.raises(ValueError):
        B.uprank(dense1, rank=3)

