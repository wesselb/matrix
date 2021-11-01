import lab as B

from matrix import Dense, Kronecker

# noinspection PyUnresolvedReferences
from ..util import (
    ConditionalContext,
    AssertDenseWarning,
    approx,
    dense1_pd,
    diag1_pd,
    kron_pd,
)


def test_cholesky_solve_diag(diag1_pd):
    chol = B.cholesky(diag1_pd)
    approx(B.cholesky_solve(chol, B.eye(chol)), B.inv(diag1_pd))


def test_cholesky_solve_lt(dense1_pd):
    chol = B.cholesky(dense1_pd)

    with AssertDenseWarning("solving <lower-triangular> x = <diagonal>"):
        approx(B.cholesky_solve(chol, B.eye(chol)), B.inv(dense1_pd))


def test_cholesky_solve_ut(dense1_pd):
    chol = B.cholesky(dense1_pd)

    with AssertDenseWarning(
        [
            "solving <upper-triangular> x = <diagonal>",
            "matrix-multiplying <upper-triangular> and <lower-triangular>",
        ]
    ):
        approx(
            B.cholesky_solve(B.transpose(chol), B.eye(chol)),
            B.inv(B.matmul(chol, chol, tr_a=True)),
        )


def test_cholesky_solve_kron(kron_pd):
    chol = B.cholesky(kron_pd)

    with ConditionalContext(
        isinstance(kron_pd.left, Dense) or isinstance(kron_pd.right, Dense),
        AssertDenseWarning("solving <lower-triangular> x = <diagonal>"),
    ):
        approx(
            B.cholesky_solve(chol, Kronecker(B.eye(chol.left), B.eye(chol.right))),
            B.inv(kron_pd),
        )
