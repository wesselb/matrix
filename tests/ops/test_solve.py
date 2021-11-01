import lab as B

from matrix import Dense, Diagonal

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_bin_op,
    dense1,
    dense2,
    dense1_pd,
    diag1,
    diag2,
    diag1_pd,
    lr1_pd,
    lt_pd,
    ut_pd,
    wb1_pd,
    zero1,
    zero2,
)


def test_solve_zero(dense1_pd, zero2):
    assert B.solve(dense1_pd, zero2) is zero2


def test_solve_dense_dense(dense1_pd, dense2):
    check_bin_op(B.solve, dense1_pd, dense2, align_dense_batch=True)


def test_solve_dense_diag(dense1_pd, diag2):
    with AssertDenseWarning("solving <dense> x = <diagonal>"):
        check_bin_op(B.solve, dense1_pd, diag2, align_dense_batch=True)


def test_solve_diag_dense(diag1_pd, dense2):
    check_bin_op(B.solve, diag1_pd, dense2, asserted_type=Dense, align_dense_batch=True)


def test_solve_diag_diag(diag1_pd, diag2):
    check_bin_op(
        B.solve, diag1_pd, diag2, asserted_type=Diagonal, align_dense_batch=True
    )


def test_solve_lt_dense(lt_pd, dense2):
    check_bin_op(B.solve, lt_pd, dense2, asserted_type=Dense, align_dense_batch=True)


def test_solve_lt_diag(lt_pd, diag2):
    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        check_bin_op(B.solve, lt_pd, diag2, asserted_type=Dense, align_dense_batch=True)


def test_solve_ut_dense(ut_pd, dense2):
    check_bin_op(B.solve, ut_pd, dense2, asserted_type=Dense, align_dense_batch=True)


def test_solve_ut_diag(ut_pd, diag2):
    with AssertDenseWarning("solving <upper-triangular> x = <diagonal>"):
        check_bin_op(B.solve, ut_pd, diag2, asserted_type=Dense, align_dense_batch=True)


def test_solve_wb_dense(wb1_pd, dense2):
    check_bin_op(B.solve, wb1_pd, dense2, align_dense_batch=True)


def test_solve_wb_diag(wb1_pd, diag2):
    check_bin_op(B.solve, wb1_pd, diag2, align_dense_batch=True)
