import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    dense_bc,
    lr1,
)


def test_squeeze_dense(dense_bc):
    check_un_op(B.squeeze, dense_bc)


def test_divide_diag_dense(lr1):
    with AssertDenseWarning("squeezing <low-rank>"):
        check_un_op(B.squeeze, lr1)
