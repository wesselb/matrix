import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1, dense2, dense1_pd, diag1_pd, lr1_pd, wb1_pd


def _check_iqf(a, b, c):
    res = B.iqf_diag(a, b, c)

    # Check correctness.
    approx(res, B.diag(B.iqf(a, b, c)))


def test_iqf_diag_two_arguments(dense1_pd, dense1):
    approx(B.iqf_diag(dense1_pd, dense1), B.iqf_diag(dense1_pd, dense1, dense1))


def test_iqf_diag_dense(dense1_pd, dense1, dense2):
    _check_iqf(dense1_pd, dense1, dense2)
    _check_iqf(dense1_pd, dense1, dense1)


def test_iqf_diag_diag(diag1_pd, dense1, dense2):
    _check_iqf(diag1_pd, dense1, dense2)
    _check_iqf(diag1_pd, dense1, dense1)


def test_iqf_diag_wb(wb1_pd, dense1, dense2):
    _check_iqf(wb1_pd, dense1, dense2)
    _check_iqf(wb1_pd, dense1, dense1)
