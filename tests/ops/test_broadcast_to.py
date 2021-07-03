import lab as B
import pytest

from matrix import Constant, Dense, Diagonal, LowRank, Zero

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, check_un_op, dense_bc, diag1, generate


def f_test(x):
    return B.broadcast_to(x, 6, 6)


@pytest.mark.parametrize(
    "zero",
    [generate(code) for code in ["zero:6,6", "zero:6,1", "zero:1,6", "zero:1,1"]],
)
def test_broadcast_to_zero(zero):
    check_un_op(f_test, zero, asserted_type=Zero)


def test_broadcast_to_dense(dense_bc):
    check_un_op(f_test, dense_bc, asserted_type=Dense)


def test_broadcast_to_diag(diag1):
    check_un_op(f_test, diag1, asserted_type=Diagonal)


@pytest.mark.parametrize(
    "const",
    [generate(code) for code in ["const:6,6", "const:6,1", "const:1,6", "const:1,1"]],
)
def test_broadcast_to_const(const):
    check_un_op(f_test, const, asserted_type=Constant)


def test_broadcast_to_lr():
    lr = LowRank(generate("dense:6,1"), generate("dense:1,1"))

    with AssertDenseWarning("broadcasting <low-rank>"):
        check_un_op(f_test, lr, asserted_type=Dense)
