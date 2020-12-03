import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, AssertDenseWarning, diag_pd, lr_pd, wb_pd


def test_pd_schur(wb_pd):
    approx(B.schur(wb_pd), B.pd_schur(wb_pd))


def test_pd_schur_wb_cache(wb_pd):
    schur1 = B.pd_schur(wb_pd)
    schur2 = B.pd_schur(wb_pd)
    assert schur1 is schur2
