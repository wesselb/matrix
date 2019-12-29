import lab as B
import numpy as np

# noinspect PyUnresolvedReferences
from ..util import allclose, dense1


def test_eye(dense1):
    allclose(B.eye(dense1), np.eye(B.shape(dense1)[0]))
