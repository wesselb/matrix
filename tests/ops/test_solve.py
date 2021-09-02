import lab as B

from matrix import Dense, Diagonal

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_bin_op,
    dense1,
    dense2,
    dense_pd,
    diag1,
    diag2,
    diag_pd,
    lr_pd,
    lt_pd,
    ut_pd,
    wb_pd,
    zero1,
    zero2,
)


def test_solve_zero(dense_pd, zero2):
    assert B.solve(dense_pd, zero2) is zero2


def test_solve_dense_dense(dense_pd, dense2):
    check_bin_op(B.solve, dense_pd, dense2)


def test_solve_dense_diag(dense_pd, diag2):
    with AssertDenseWarning("solving <dense> x = <diagonal>"):
        check_bin_op(B.solve, dense_pd, diag2)


def test_solve_diag_dense(diag_pd, dense2):
    check_bin_op(B.solve, diag_pd, dense2, asserted_type=Dense)


def test_solve_diag_diag(diag_pd, diag2):
    check_bin_op(B.solve, diag_pd, diag2, asserted_type=Diagonal)


def test_solve_lt_dense(lt_pd, dense2):
    check_bin_op(B.solve, lt_pd, dense2, asserted_type=Dense)


def test_solve_lt_diag(lt_pd, diag2):
    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        check_bin_op(B.solve, lt_pd, diag2, asserted_type=Dense)


def test_solve_ut_dense(ut_pd, dense2):
    check_bin_op(B.solve, ut_pd, dense2, asserted_type=Dense)


def test_solve_ut_diag(ut_pd, diag2):
    with AssertDenseWarning("solving <upper-triangular> x = <diagonal>"):
        check_bin_op(B.solve, ut_pd, diag2, asserted_type=Dense)


def test_solve_wb_dense(wb_pd, dense2):
    check_bin_op(B.solve, wb_pd, dense2)


def test_solve_wb_diag(wb_pd, diag2):
    check_bin_op(B.solve, wb_pd, diag2)
