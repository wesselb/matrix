import lab as B
from plum import convert, promote

from matrix import LowRank, Zero, Constant, Dense
# noinspection PyUnresolvedReferences
from .util import approx, const1, const_r, dense1


def test_promote_numeric_zero(dense1):
    assert isinstance(promote(dense1, 0)[1], Zero)
    assert isinstance(promote(dense1, 0.0)[1], Zero)


def test_promote_numeric_scalar(dense1):
    _, res = promote(dense1, 1)
    assert isinstance(res, Constant)
    assert B.shape(res) == (1, 1)


def test_promote_numeric_vector(dense1):
    _, res = promote(dense1, B.ones(3))
    assert isinstance(res, Dense)
    assert B.shape(res) == (3, 1)


def test_promote_numeric_matrix(dense1):
    _, res = promote(dense1, B.ones(3, 4))
    assert isinstance(res, Dense)
    assert B.shape(res) == (3, 4)


def test_promote_numeric_matrix_batch(dense1):
    _, res = promote(dense1, B.ones(2, 3, 4))
    assert isinstance(res, Dense)
    assert B.shape_batch(res) == (2,)
    assert B.shape_matrix(res) == (3, 4)


def test_constant_to_lowrank_square(const1):
    res = convert(const1, LowRank)
    approx(const1, res)

    assert isinstance(res, LowRank)
    assert res.left is res.right


def test_constant_to_lowrank_rectangular(const_r):
    res = convert(const_r, LowRank)
    approx(const_r, res)

    assert isinstance(res, LowRank)
