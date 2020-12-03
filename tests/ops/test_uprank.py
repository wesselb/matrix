import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1


def test_uprank_dense(dense1):
    assert B.uprank(dense1) is dense1
