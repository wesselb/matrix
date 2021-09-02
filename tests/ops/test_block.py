import lab as B

from matrix import Dense, Diagonal, Zero

from ..util import AssertDenseWarning, approx, generate


def check_block(res, rows):
    approx(res, B.concat2d(*[[B.dense(x) for x in row] for row in rows]))


def test_block_one_block():
    rows = [[generate("dense:3,3")]]
    assert B.block(*rows) is rows[0][0]


def test_block_dense():
    rows = [[generate("dense:6,6") for _ in range(3)] for _ in range(3)]
    with AssertDenseWarning("could not preserve structure"):
        res = B.block(*rows)


def test_block_zero():
    rows = [[generate("zero:6,6") for _ in range(3)] for _ in range(3)]
    res = B.block(*rows)
    check_block(res, rows)
    assert isinstance(res, Zero)


def test_block_diag():
    rows = [
        [generate("diag:6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:6,6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:3,6"), generate("zero:3,6"), generate("diag:3")],
    ]
    res = B.block(*rows)
    check_block(res, rows)
    assert isinstance(res, Diagonal)


def test_block_diag_offdiagonal_check():
    rows = [
        [generate("diag:6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:6,6"), generate("zero:6,6"), generate("randn:6,3")],
        [generate("zero:3,6"), generate("zero:3,6"), generate("diag:3")],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)


def test_block_diag_diagonal_check():
    rows = [
        [generate("diag:6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:6,6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:3,6"), generate("zero:3,6"), generate("randn:3,3")],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)


def test_block_diag_square_check():
    rows = [
        [generate("diag:6"), generate("zero:6,7"), generate("zero:6,3")],
        [generate("zero:6,6"), generate("zero:6,7"), generate("zero:6,3")],
        [generate("zero:3,6"), generate("zero:3,7"), generate("diag:3")],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)

    rows = [
        [generate("diag:6"), generate("zero:6,6"), generate("zero:6,6")],
        [generate("zero:6,6"), generate("zero:6,6"), generate("zero:6,6")],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)
