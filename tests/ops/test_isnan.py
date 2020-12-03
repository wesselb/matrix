import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, AssertDenseWarning, dense1, diag1


def test_take_dense(dense1):
    check_un_op(B.isnan, dense1)


def test_take_diag(diag1):
    with AssertDenseWarning('applying "isnan" to <diagonal>'):
        check_un_op(B.isnan, diag1)
