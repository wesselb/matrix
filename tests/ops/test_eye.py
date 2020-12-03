import lab as B
import numpy as np

from matrix import Diagonal

# noinspection PyUnresolvedReferences
from ..util import approx, dense1


def test_eye(dense1):
    approx(B.eye(dense1), np.eye(B.shape(dense1)[0]))
    assert isinstance(B.eye(dense1), Diagonal)
