import lab as B
from matrix import structured
from matrix.ops.util import align_batch

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    ConditionalContext,
    approx,
    check_un_op,
    dense1,
    dense2,
    dense1_pd,
    dense2_pd,
    diag1_pd,
    diag2_pd,
    lr1_pd,
    lr2_pd,
    wb1_pd,
    wb2_pd,
)


def _check_ratio(a, b):
    res = B.ratio(a, b)

    # Check correctness.
    a_dense, b_dense = align_batch(B.dense(a), B.dense(b))
    approx(res, B.trace(B.solve(b_dense, a_dense)))


def test_ratio_dense_diag(dense1_pd, diag2_pd):
    _check_ratio(dense1_pd, diag2_pd)


def test_ratio_diag_dense(diag1_pd, dense2_pd):
    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        _check_ratio(diag1_pd, dense2_pd)


def test_ratio_dense_wb(dense1_pd, wb2_pd):
    with ConditionalContext(
        structured(wb2_pd.lr.left) or structured(wb2_pd.lr.right),
        AssertDenseWarning("getting the diagonal of <low-rank>"),
    ):
        _check_ratio(dense1_pd, wb2_pd)


def test_ratio_wb_dense(wb1_pd, dense2_pd):
    warns = [
        "solving <lower-triangular> x = <lower-triangular>",
        "converting <woodbury>",
    ]
    with AssertDenseWarning(warns):
        _check_ratio(wb1_pd, dense2_pd)


def test_ratio_wb_wb(wb1_pd, wb2_pd):
    with ConditionalContext(
        (structured(wb1_pd.lr.left) or structured(wb1_pd.lr.right))
        or (structured(wb2_pd.lr.left) or structured(wb2_pd.lr.right)),
        AssertDenseWarning("getting the diagonal of <low-rank>"),
    ):
        _check_ratio(wb1_pd, wb2_pd)
