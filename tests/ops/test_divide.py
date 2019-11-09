import lab as B

from matrix import Dense
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_op,

    diag1,
    diag2,
    dense1,
    dense2
)


def test_multiply_dense(dense1, dense2):
    check_op(B.divide, dense1, dense2, asserted_type=Dense)
