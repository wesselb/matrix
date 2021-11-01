import lab as B

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, check_un_op, diag1_pd, lr1_pd, wb1_pd

# Correctness is tested in `logdet` et cetera.


def test_schur_wb_cache(wb1_pd):
    schur1 = B.schur(wb1_pd)
    schur2 = B.schur(wb1_pd)
    assert schur1 is schur2
