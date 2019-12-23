from matrix import Constant, Zero


def test_constant_formatting():
    assert str(Constant(1, 3, 3)) == \
           '<constant matrix: shape=3x3, dtype=int, const=1>'
    assert repr(Constant(1, 3, 3)) == \
           '<constant matrix: shape=3x3, dtype=int, const=1>'


def test_zero_formatting():
    assert str(Zero(int, 3, 3)) == '<zero matrix: shape=3x3, dtype=int>'
    assert repr(Zero(int, 3, 3)) == '<zero matrix: shape=3x3, dtype=int>'
