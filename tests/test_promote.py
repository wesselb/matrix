import lab as B
import pytest
from plum import promote, convert

from matrix import Zero, LowRank

# noinspection PyUnresolvedReferences
from .util import approx, dense1, const1, const_r


def test_promote_check(dense1):
    with pytest.raises(RuntimeError):
        promote(B.ones(3), dense1)


def test_promote_zero(dense1):
    assert isinstance(promote(dense1, 0)[1], Zero)
    assert isinstance(promote(dense1, 0.0)[1], Zero)


def test_constant_to_lowrank_square(const1):
    res = convert(const1, LowRank)
    approx(const1, res)

    assert isinstance(res, LowRank)
    assert res.left is res.right


def test_constant_to_lowrank_rectangular(const_r):
    res = convert(const_r, LowRank)
    approx(const_r, res)

    assert isinstance(res, LowRank)
