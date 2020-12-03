import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    zero1,
    dense1,
    diag1,
    const1,
    lr1,
    wb1,
    kron1,
)


def _reshape(a):
    rows, cols = B.shape(a)
    return B.reshape(a, rows * cols, -1)


def test_reshape_dense(dense1):
    check_un_op(_reshape, dense1)


def test_reshape_diag(diag1):
    with AssertDenseWarning("converting <diagonal> to dense for reshaping"):
        check_un_op(_reshape, diag1)
