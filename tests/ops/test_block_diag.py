import lab as B
from scipy.linalg import block_diag

from matrix import Dense, Diagonal, TiledBlocks, Zero

from ..util import AssertDenseWarning, approx, generate


def check_block_diag(res, blocks):
    approx(res, block_diag(*(B.dense(block) for block in blocks)))


def test_block_diag_one_block():
    blocks = [generate("dense:3,3")]
    assert B.block_diag(*blocks) is blocks[0]


def test_block_diag_dense():
    blocks = [generate("dense:3,3") for _ in range(3)]
    with AssertDenseWarning("Could not preserve structure in block-diagonal matrix"):
        res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Dense)


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


def test_block_diag_tiledblocks():
    tb = TiledBlocks(
        (generate("diag:3"), 2),
        (generate("dense:3,3"), 1),
        (generate("zero:3,3"), 2),
    )
    blocks = [
        tb.blocks[0],
        tb.blocks[0],
        tb.blocks[1],
        tb.blocks[2],
        tb.blocks[2],
    ]
    with AssertDenseWarning("Could not preserve structure in block-diagonal matrix"):
        res = B.block_diag(tb)
    check_block_diag(res, blocks)
