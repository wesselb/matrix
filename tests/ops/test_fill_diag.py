import lab as B
import numpy as np

from ..util import approx


def test_fill_diag():
    approx(B.fill_diag(2, 3), np.diag(np.array([2, 2, 2])))
