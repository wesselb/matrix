import lab as B
from scipy.linalg import block_diag

from matrix import Dense, Zero, Diagonal
from ..util import approx, generate, AssertDenseWarning


def check_block_diag(res, blocks):
    approx(res, block_diag(*(B.dense(block) for block in blocks)))


def test_block_diag_zero():
    blocks = [generate("zero:3,3") for _ in range(3)]
    res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Zero)


def test_block_diag_diag():
    blocks = [generate("diag:3") for _ in range(3)]
    res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Diagonal)


def test_block_diag_dense():
    blocks = [generate("dense:3,3") for _ in range(3)]
    with AssertDenseWarning("Could not preserve structure in block matrix"):
        res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Dense)
