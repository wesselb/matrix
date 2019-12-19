from matrix import Constant, Zero


def test_constant_str():
    assert str(Constant(1, 3, 3)) == \
           '<constant matrix: shape=3x3, dtype=int, const=1>'


def test_zero_str():
    assert str(Zero(3, 3)) == '<zero matrix: shape=3x3>'
