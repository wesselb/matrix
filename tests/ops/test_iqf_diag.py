import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense1,
    dense2,
    dense_pd,
    diag_pd,
    lr_pd,
    wb_pd
)


def _check_iqf(a, b, c):
    res = B.iqf_diag(a, b, c)

    # Check correctness.
    allclose(res, B.diag(B.iqf(a, b, c)))


def test_iqf_diag_two_arguments(dense_pd, dense1):
    allclose(B.iqf_diag(dense_pd, dense1), B.iqf_diag(dense_pd, dense1, dense1))


def test_iqf_diag_dense(dense_pd, dense1, dense2):
    _check_iqf(dense_pd, dense1, dense2)
    _check_iqf(dense_pd, dense1, dense1)


def test_iqf_diag_diag(diag_pd, dense1, dense2):
    _check_iqf(diag_pd, dense1, dense2)
    _check_iqf(diag_pd, dense1, dense1)


def test_iqf_diag_wb(wb_pd, dense1, dense2):
    _check_iqf(wb_pd, dense1, dense2)
    _check_iqf(wb_pd, dense1, dense1)
