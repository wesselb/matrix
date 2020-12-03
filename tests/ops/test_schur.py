import lab as B

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, AssertDenseWarning, diag_pd, lr_pd, wb_pd


# Correctness is tested in `logdet` et cetera.


def test_schur_wb_cache(wb_pd):
    schur1 = B.schur(wb_pd)
    schur2 = B.schur(wb_pd)
    assert schur1 is schur2
