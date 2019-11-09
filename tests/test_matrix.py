# noinspection PyUnresolvedReferences
from .util import dense1, dense2
from matrix import Dense


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
