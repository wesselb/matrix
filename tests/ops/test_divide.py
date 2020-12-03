import lab as B

from matrix import Dense

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_bin_op,
    AssertDenseWarning,
    dense1,
    dense2,
    diag1,
    diag2,
)


def test_divide_dense(dense1, dense2):
    check_bin_op(B.divide, dense1, dense2, asserted_type=Dense)


def test_divide_diag_dense(diag1, dense2):
    with AssertDenseWarning("dividing <diagonal> by <dense>"):
        check_bin_op(B.divide, diag1, dense2, asserted_type=Dense)
