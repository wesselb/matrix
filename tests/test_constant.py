from matrix import Constant, Zero


def test_constant_str():
    assert str(Constant(1, 3, 3)) == \
           'Constant 3x3 matrix of data type int with constant 1'


def test_zero_str():
    assert str(Zero(3, 3)) == 'Constant 3x3 matrix of zeros'
