import lab as B
import numpy as np

from matrix import Dense
# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    zero1,
    dense1,
    diag1,
    const1,
    lr1,
    wb1,
    kron1,
)


def _reshape_to_matrix(a):
    rows, cols = B.shape(a)
    return B.reshape(a, rows * cols, -1)


def _reshape_to_vector(a):
    rows, cols = B.shape(a)
    return B.reshape(a, rows * cols)


def test_reshape_dense(dense1):
    check_un_op(_reshape_to_matrix, dense1, asserted_type=Dense)
    check_un_op(_reshape_to_vector, dense1, asserted_type=np.ndarray)


def test_reshape_diag(diag1):
    with AssertDenseWarning("converting <diagonal> to dense for reshaping"):
        check_un_op(_reshape_to_matrix, diag1, asserted_type=Dense)
        check_un_op(_reshape_to_vector, diag1, asserted_type=np.ndarray)
