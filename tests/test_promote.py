import lab as B
import pytest
from plum import promote

from matrix import Zero
# noinspection PyUnresolvedReferences
from .util import dense1


def test_promote_check(dense1):
    with pytest.raises(RuntimeError):
        promote(B.ones(3), dense1)


def test_promote_zero(dense1):
    assert isinstance(promote(dense1, 0)[1], Zero)
    assert isinstance(promote(dense1, 0.0)[1], Zero)
