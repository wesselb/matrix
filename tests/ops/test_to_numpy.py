import lab as B
from plum import isinstance

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, check_un_op, dense1, diag1


def test_to_numpy_dense(dense1):
    assert isinstance(B.to_numpy(dense1), B.NP)
    approx(B.to_numpy(dense1), B.dense(dense1))


def test_to_numpy_diag(diag1):
    assert isinstance(B.to_numpy(diag1), B.NP)
    approx(B.to_numpy(diag1), B.dense(diag1))
