import lab as B

# noinspection PyUnresolvedReferences
from ..util import allclose, check_un_op, dense1


def test_trace_dense(dense1):
    check_un_op(B.trace, dense1)
