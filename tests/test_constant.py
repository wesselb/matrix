import lab as B
import pytest

from matrix import Zero, Constant


def test_zero_formatting():
    assert str(Zero(int, 3, 3)) == "<zero matrix: batch=(), shape=(3, 3), dtype=int>"
    assert repr(Zero(int, 3, 3)) == "<zero matrix: batch=(), shape=(3, 3), dtype=int>"


def test_zero_attributes():
    zero = Zero(int, 3, 4)
    assert zero.dtype == int
    assert zero.batch == ()
    assert zero.rows == 3
    assert zero.cols == 4

    # Test caching of `dense`.
    assert zero.dense is None
    B.dense(zero)
    assert zero.dense is not None

    zero = Zero(int, 2, 3, 4)
    assert zero.dtype == int
    assert zero.batch == (2,)
    assert zero.rows == 3
    assert zero.cols == 4

    # Test caching of `dense`.
    assert zero.dense is None
    B.dense(zero)
    assert zero.dense is not None


def test_zero_checks():
    with pytest.raises(ValueError):
        Zero(int)


def test_constant_formatting():
    assert (
        str(Constant(1, 3, 3)) == "<constant matrix: batch=(), shape=(3, 3), dtype=int>"
    )
    assert (
        repr(Constant(1, 3, 3)) == ""
        "<constant matrix: batch=(), shape=(3, 3), dtype=int\n"
        " const=1>"
    )


def test_constant_attributes():
    const = Constant(1, 3, 4)
    assert const.const == 1
    assert const.rows == 3
    assert const.cols == 4

    # Test caching of `dense`.
    assert const.dense is None
    B.dense(const)
    assert const.dense is not None

    # Test caching of `cholesky`.
    const = Constant(1, 3, 3)
    assert const.cholesky is None
    B.cholesky(const)
    assert const.cholesky is not None


def test_constant_checks():
    with pytest.raises(ValueError):
        Constant(1)
    with pytest.raises(ValueError):
        Constant(1, 1)


def test_constant_batch_inference():
    const = Constant(1, 4, 5)
    assert B.shape(const.const) == ()
    assert B.shape_batch(const) == ()
    assert B.shape_matrix(const) == (4, 5)

    const = Constant(1, 2, 4, 5)
    assert B.shape(const.const) == (2,)
    assert B.shape_batch(const) == (2,)
    assert B.shape_matrix(const) == (4, 5)

    const = Constant(B.ones(2), 2, 4, 5)
    assert B.shape(const.const) == (2,)
    assert B.shape_batch(const) == (2,)
    assert B.shape_matrix(const) == (4, 5)

    const = Constant(B.ones(2), 4, 5)
    assert B.shape(const.const) == (2,)
    assert B.shape_batch(const) == (2,)
    assert B.shape_matrix(const) == (4, 5)
