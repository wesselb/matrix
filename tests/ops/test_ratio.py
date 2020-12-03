import lab as B

from matrix import structured

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    ConditionalContext,
    dense1,
    dense2,
    dense_pd,
    diag_pd,
    lr_pd,
    wb_pd,
)


def _check_ratio(a, b):
    res = B.ratio(a, b)

    # Check correctness.
    approx(res, B.trace(B.solve(B.dense(b), B.dense(a))))


def test_ratio_dense_diag(dense_pd, diag_pd):
    _check_ratio(dense_pd, diag_pd)


def test_ratio_diag_dense(diag_pd, dense_pd):
    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        _check_ratio(diag_pd, dense_pd)


def test_ratio_dense_wb(dense_pd, wb_pd):
    with AssertDenseWarning("adding <lower-triangular> and <low-rank>"):
        _check_ratio(dense_pd, wb_pd)


def test_ratio_wb_dense(wb_pd, dense_pd):
    warns = [
        "solving <lower-triangular> x = <lower-triangular>",
        "converting <woodbury>",
    ]
    with AssertDenseWarning(warns):
        _check_ratio(wb_pd, dense_pd)
