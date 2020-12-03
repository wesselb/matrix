import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1


def test_one(dense1):
    check_un_op(B.one, dense1)
