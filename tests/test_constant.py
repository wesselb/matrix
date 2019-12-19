from matrix import Constant


def test_dense_str():
    assert str(Constant(1, 3, 3)) == \
           'Constant 3x3 matrix of data type int with constant 1'
