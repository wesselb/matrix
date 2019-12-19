import lab as B
import pytest
from plum import promote

# noinspection PyUnresolvedReferences
from .util import dense1


def test_promote_check(dense1):
    with pytest.raises(RuntimeError):
        promote(B.ones(3), dense1)
