import lab as B
import pytest

from matrix import Constant, Dense, Diagonal, LowRank, Zero

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, check_un_op, dense_bc, diag1, generate


def _broadcast_matrix(x):
    return B.broadcast_to(x, *B.shape_batch(x), 6, 6)


def _broadcast_batch(x):
    return B.broadcast_to(x, 3, 3, *B.shape(x))


@pytest.mark.parametrize(
    "zero",
    map(generate, ["|zero:6,6", "|zero:6,1", "|zero:1,6", "|zero:1,1"]),
)
def test_broadcast_to_zero(zero):
    check_un_op(_broadcast_batch, zero, asserted_type=Zero)
    check_un_op(_broadcast_matrix, zero, asserted_type=Zero)


def test_broadcast_to_dense(dense_bc):
    check_un_op(_broadcast_batch, dense_bc, asserted_type=Dense)
    check_un_op(_broadcast_matrix, dense_bc, asserted_type=Dense)


def test_broadcast_to_diag(diag1):
    check_un_op(_broadcast_batch, diag1, asserted_type=Diagonal)
    check_un_op(_broadcast_matrix, diag1, asserted_type=Diagonal)


@pytest.mark.parametrize(
    "const",
    map(generate, ["|const:6,6", "|const:6,1", "|const:1,6", "|const:1,1"]),
)
def test_broadcast_to_const(const):
    check_un_op(_broadcast_batch, const, asserted_type=Constant)
    check_un_op(_broadcast_matrix, const, asserted_type=Constant)


def test_broadcast_to_lr():
    lr = LowRank(generate("|dense:6,1"), generate("|dense:1,1"))

    check_un_op(_broadcast_batch, lr, asserted_type=LowRank)
    with AssertDenseWarning("broadcasting <low-rank>"):
        check_un_op(_broadcast_matrix, lr, asserted_type=Dense)
