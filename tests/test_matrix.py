import lab as B

from matrix import Dense, Diagonal, structured

# noinspection PyUnresolvedReferences
from .util import approx, AssertDenseWarning, dense1, dense2, diag1


def test_dispatch(dense1, dense2):
    assert isinstance(-dense1, Dense)
    assert isinstance(dense1 + dense2, Dense)
    assert isinstance(dense1.__radd__(dense2), Dense)
    assert isinstance(dense1 - dense2, Dense)
    assert isinstance(dense1.__rsub__(dense2), Dense)
    assert isinstance(dense1 * dense2, Dense)
    assert isinstance(dense1.__rmul__(dense2), Dense)
    assert isinstance(dense1 / dense2, Dense)
    assert isinstance(dense1.__rtruediv__(dense2), Dense)
    assert isinstance(dense1 ** 2, Dense)
    assert isinstance(dense1 @ dense2, Dense)


def test_properties(dense1):
    approx(dense1.T, B.transpose(dense1))
    assert dense1.shape == B.shape(dense1)
    assert dense1.dtype == B.dtype(dense1)


def test_dense_formatting():
    assert str(Dense(B.ones(3, 3))) == "<dense matrix: shape=3x3, dtype=float64>"
    assert (
        repr(Dense(B.ones(3, 3))) == "<dense matrix: shape=3x3, dtype=float64\n"
        " mat=[[1. 1. 1.]\n"
        "      [1. 1. 1.]\n"
        "      [1. 1. 1.]]>"
    )


def test_dense_attributes():
    mat = B.ones(3, 3)
    dense = Dense(mat)
    assert dense.mat is mat


def test_structured():
    assert structured(Diagonal(B.ones(3)))
    assert not structured(Dense(B.ones(3, 3)))
    assert not structured(B.ones(3, 3))


def test_conversion_to_dense(diag1):
    approx(Dense(diag1), diag1)
    assert isinstance(Dense(diag1).mat, B.Numeric)


def test_indexing_dense(dense1):
    approx(dense1[2], B.dense(dense1)[2])


def test_indexing_diag(diag1):
    with AssertDenseWarning("indexing into <diagonal>"):
        approx(diag1[2], B.dense(diag1)[2])
