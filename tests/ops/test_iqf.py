import lab as B

from matrix.ops.util import align_batch

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    dense1,
    dense2,
    dense_pd,
    diag_pd,
    lr_pd,
    wb_pd,
)


def _check_iqf(a, b, c):
    res = B.iqf(a, b, c)

    # Check correctness.
    a_dense, c_dense = align_batch(a, c)
    approx(res, B.mm(B.dense(b), B.solve(a_dense, c_dense), tr_a=True))


def test_iqf_two_arguments(dense_pd, dense1):
    approx(B.iqf(dense_pd, dense1), B.iqf(dense_pd, dense1, dense1))


def test_iqf_dense(dense_pd, dense1, dense2):
    _check_iqf(dense_pd, dense1, dense2)
    _check_iqf(dense_pd, dense1, dense1)


def test_iqf_diag(diag_pd, dense1, dense2):
    _check_iqf(diag_pd, dense1, dense2)
    _check_iqf(diag_pd, dense1, dense1)


def test_iqf_wb(wb_pd, dense1, dense2):
    _check_iqf(wb_pd, dense1, dense2)
    _check_iqf(wb_pd, dense1, dense1)
