import lab as B

# noinspection PyUnresolvedReferences
from ..util import allclose, check_bin_op, AssertDenseWarning, dense_pd


def test_cholesky_solve_lt(dense_pd):
    chol = B.cholesky(dense_pd)

    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        allclose(B.cholesky_solve(chol, B.eye(chol)), B.inv(dense_pd))


def test_cholesky_solve_ut(dense_pd):
    chol = B.cholesky(dense_pd)

    with AssertDenseWarning(
        [
            "solving <upper-triangular> x = <diagonal>",
            "matrix-multiplying <upper-triangular> and <lower-triangular>",
        ]
    ):
        allclose(
            B.cholesky_solve(B.transpose(chol), B.eye(chol)),
            B.inv(B.matmul(chol, chol, tr_a=True)),
        )
