import lab as B
import numpy as np

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const1,
    dense1,
    diag1,
    kron1,
    lr1,
    wb1,
    zero1,
)


def _reshape_to_vector(a):
    rows, cols = B.shape_matrix(a)
    return B.reshape(a, *B.shape_batch(a), rows * cols)


def test_reshape_dense(dense1):
    check_un_op(_reshape_to_vector, dense1, asserted_type=np.ndarray)


def test_reshape_diag(diag1):
    with AssertDenseWarning("converting <diagonal> to dense for reshaping"):
        check_un_op(_reshape_to_vector, diag1, asserted_type=np.ndarray)
