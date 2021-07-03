import lab.jax as B

from matrix import Constant, Dense, Zero

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, check_un_op, const1, dense1, diag1, zero1


def test_tile_zero(zero1):
    check_un_op(lambda x: B.tile(x, 2, 2), zero1, asserted_type=Zero)


def test_tile_dense(dense1):
    check_un_op(lambda x: B.tile(x, 2, 2), dense1, asserted_type=Dense)


def test_tile_diag(diag1):
    with AssertDenseWarning("tiling <diagonal>"):
        check_un_op(lambda x: B.tile(x, 2, 2), diag1, asserted_type=Dense)


def test_tile_const(const1):
    check_un_op(lambda x: B.tile(x, 2, 2), const1, asserted_type=Constant)
