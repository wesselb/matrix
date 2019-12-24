# noinspection PyUnresolvedReferences
from .util import allclose, dense1, dense2
from matrix import Dense
import lab as B


def test_dispatch(dense1, dense2):
    assert isinstance(-dense1, Dense)
    assert isinstance(dense1 + dense2, Dense)
    assert isinstance(dense1.__radd__(dense2), Dense)
    assert isinstance(dense1 - dense2, Dense)
    assert isinstance(dense1.__rsub__(dense2), Dense)
    assert isinstance(dense1 * dense2, Dense)
    assert isinstance(dense1.__rmul__(dense2), Dense)
    assert isinstance(dense1 / dense2, Dense)
    assert isinstance(dense1.__rtruediv__(dense2), Dense)
    assert isinstance(dense1 ** 2, Dense)
    assert isinstance(dense1 @ dense2, Dense)


def test_properties(dense1):
    allclose(dense1.T, B.transpose(dense1))
    assert dense1.shape == B.shape(dense1)
    assert dense1.dtype == B.dtype(dense1)


def test_dense_formatting():
    assert str(Dense(B.ones(3, 3))) == \
           '<dense matrix: shape=3x3, dtype=float64>'
    assert repr(Dense(B.ones(3, 3))) == \
           '<dense matrix: shape=3x3, dtype=float64\n' \
           ' mat=[[1. 1. 1.]\n' \
           '      [1. 1. 1.]\n' \
           '      [1. 1. 1.]]>'


def test_dense_attributes():
    mat = B.ones(3, 3)
    dense = Dense(mat)
    assert dense.mat is mat
