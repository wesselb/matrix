import lab as B

from matrix import Zero

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1


def test_zeros(dense1):
    check_un_op(B.zeros, dense1, asserted_type=Zero)
