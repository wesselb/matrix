import lab as B

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, check_un_op, diag1_pd, lr1_pd, wb1_pd


def test_pd_schur(wb1_pd):
    approx(B.schur(wb1_pd), B.pd_schur(wb1_pd))


def test_pd_schur_wb_cache(wb1_pd):
    schur1 = B.pd_schur(wb1_pd)
    schur2 = B.pd_schur(wb1_pd)
    assert schur1 is schur2
