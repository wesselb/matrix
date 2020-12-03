import lab as B

from matrix import Constant

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1


def test_ones(dense1):
    check_un_op(B.ones, dense1, asserted_type=Constant)
