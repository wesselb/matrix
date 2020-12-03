import lab as B

from matrix import Dense, Zero, Diagonal
from ..util import approx, generate, AssertDenseWarning


def _dense(rows):
    return [[B.dense(x) for x in row] for row in rows]


def test_block_zero():
    rows = [[generate("zero:6,6") for _ in range(3)] for _ in range(3)]
    res = B.block(*rows)
    approx(res, B.concat2d(*_dense(rows)))
    assert isinstance(res, Zero)


def test_block_dense():
    rows = [[generate("dense:6,6") for _ in range(3)] for _ in range(3)]
    with AssertDenseWarning("could not preserve structure"):
        res = B.block(*rows)
    approx(res, B.concat2d(*_dense(rows)))


def test_block_diag():
    rows = [
        [generate("diag:6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:6,6"), generate("zero:6,6"), generate("zero:6,3")],
        [generate("zero:3,6"), generate("zero:3,6"), generate("diag:3")],
    ]
    res = B.block(*rows)
    approx(res, B.concat2d(*_dense(rows)))
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
