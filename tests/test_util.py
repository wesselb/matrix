import numpy as np
from matrix.util import dtype_str


def test_dtype_str():
    assert dtype_str(np.float64) == "float64"
    assert dtype_str(np.dtype("float64")) == "float64"
