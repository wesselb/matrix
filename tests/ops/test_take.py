import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, AssertDenseWarning, dense1, diag1


def _check_take(a):
    approx(B.take(a, [0, 1]), B.take(B.dense(a), [0, 1]))


def test_take_dense(dense1):
    _check_take(dense1)


def test_take_diag(diag1):
    with AssertDenseWarning("taking from <diagonal>"):
        _check_take(diag1)
