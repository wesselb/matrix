# noinspection PyUnresolvedReferences
from .util import dense1, dense2
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


def test_repr():
    assert str(Dense(B.ones(3))) == repr(Dense(B.ones(3)))


def test_dense_str():
    assert str(Dense(B.ones(3))) == 'Dense matrix:\n  [1. 1. 1.]'
