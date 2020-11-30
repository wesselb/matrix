from matrix import Zero, Constant


def test_zero_formatting():
    assert str(Zero(int, 3, 3)) == "<zero matrix: shape=3x3, dtype=int>"
    assert repr(Zero(int, 3, 3)) == "<zero matrix: shape=3x3, dtype=int>"


def test_zero_attributes():
    zero = Zero(int, 3, 4)
    assert zero.dtype == int
    assert zero.rows == 3
    assert zero.cols == 4


def test_constant_formatting():
    assert str(Constant(1, 3, 3)) == "<constant matrix: shape=3x3, dtype=int, const=1>"
    assert repr(Constant(1, 3, 3)) == "<constant matrix: shape=3x3, dtype=int, const=1>"


def test_constant_attributes():
    const = Constant(1, 3, 4)
    assert const.const == 1
    assert const.rows == 3
    assert const.cols == 4
