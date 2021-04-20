import lab as B

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
