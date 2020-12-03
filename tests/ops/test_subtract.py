import lab as B

from matrix import Dense

# noinspection PyUnresolvedReferences
from ..util import approx, check_bin_op, dense1, dense2


def test_subtract_dense(dense1, dense2):
    check_bin_op(B.subtract, dense1, dense2, asserted_type=Dense)
