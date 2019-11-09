import pytest
import lab as B
from numpy.testing import assert_allclose, assert_array_almost_equal

__all__ = ['allclose',
           'approx',

           # Fixtures:
           'mat',
           'vec']

allclose = assert_allclose
approx = assert_array_almost_equal


# Fixtures:

@pytest.fixture(params=[(3, 3), (4, 3), (3, 4)])
def mat(request):
    return B.randn(*request.param)


@pytest.fixture()
def vec():
    yield B.randn(3)
