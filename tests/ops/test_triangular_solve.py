import lab as B
import pytest

from matrix import Dense

# noinspection PyUnresolvedReferences
from ..util import approx, check_bin_op, AssertDenseWarning, lt_pd, ut_pd, dense2


def test_cholesky_solve_lt(lt_pd, dense2):
    check_bin_op(
        B.triangular_solve, lt_pd, dense2, asserted_type=Dense, check_broadcasting=False
    )

    with pytest.warns(UserWarning):
        B.triangular_solve(B.transpose(lt_pd), dense2)


def test_cholesky_solve_ut(ut_pd, dense2):
    check_bin_op(
        lambda a, b: B.triangular_solve(a, b, lower_a=False),
        ut_pd,
        dense2,
        asserted_type=Dense,
        check_broadcasting=False,
    )

    with pytest.warns(UserWarning):
        B.triangular_solve(B.transpose(ut_pd), dense2, lower_a=False)
