import lab as B
from scipy.linalg import block_diag

from matrix import Dense, Diagonal, TiledBlocks, Zero

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, generate, gen_batch_code as gen


def check_block_diag(res, blocks):
    # `scipy`s `block_diag` does not work with batches. If there are batches, just
    # check the shape.
    if all(B.shape_batch(block) == () for block in blocks):
        approx(res, block_diag(*(B.dense(block) for block in blocks)))
    else:
        rows, cols = map(sum, zip(*(B.shape_matrix(block) for block in blocks)))
        assert B.shape(res) == B.shape_batch_broadcast(*blocks) + (rows, cols)


def test_block_diag_one_block(gen):
    blocks = [generate(f"{gen()}dense:3,3")]
    assert B.block_diag(*blocks) is blocks[0]


def test_block_diag_dense(gen):
    blocks = [generate(f"{gen()}dense:3,3") for _ in range(3)]
    with AssertDenseWarning("Could not preserve structure in block-diagonal matrix"):
        res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Dense)


def test_block_diag_zero(gen):
    blocks = [generate(f"{gen()}zero:3,3") for _ in range(3)]
    res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Zero)


def test_block_diag_diag(gen):
    blocks = [generate(f"{gen()}diag:3") for _ in range(3)]
    res = B.block_diag(*blocks)
    check_block_diag(res, blocks)
    assert isinstance(res, Diagonal)


def test_block_diag_tiledblocks(gen):
    tb = TiledBlocks(
        (generate(f"{gen()}diag:3"), 2),
        (generate(f"{gen()}dense:3,3"), 1),
        (generate(f"{gen()}zero:3,3"), 2),
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
