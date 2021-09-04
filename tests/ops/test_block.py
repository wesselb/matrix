import lab as B

from matrix import Dense, Diagonal, Zero
from matrix.ops.util import align_batch

# noinspection PyUnresolvedReferences
from ..util import AssertDenseWarning, approx, generate, gen_batch_code as gen


def check_block(res, rows):
    rows = [[B.dense(x) for x in row] for row in rows]
    # Align the batch dimensions before concatenating.
    approx(res, B.concat2d(*align_batch(*rows)))


def test_block_one_block(gen):
    rows = [[generate(f"{gen()}dense:3,3")]]
    assert B.block(*rows) is rows[0][0]


def test_block_dense(gen):
    rows = [[generate(f"{gen()}dense:6,6") for _ in range(3)] for _ in range(3)]
    with AssertDenseWarning("could not preserve structure"):
        B.block(*rows)


def test_block_zero(gen):
    rows = [[generate(f"{gen()}zero:6,6") for _ in range(3)] for _ in range(3)]
    res = B.block(*rows)
    check_block(res, rows)
    assert isinstance(res, Zero)


def test_block_diag(gen):
    rows = [
        [
            generate(f"{gen()}diag:6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}diag:3"),
        ],
    ]
    res = B.block(*rows)
    check_block(res, rows)
    assert isinstance(res, Diagonal)


def test_block_diag_offdiagonal_check(gen):
    rows = [
        [
            generate(f"{gen()}diag:6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}randn:6,3"),
        ],
        [
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}diag:3"),
        ],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)


def test_block_diag_diagonal_check(gen):
    rows = [
        [
            generate(f"{gen()}diag:6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}randn:3,3"),
        ],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)


def test_block_diag_square_check(gen):
    rows = [
        [
            generate(f"{gen()}diag:6"),
            generate(f"{gen()}zero:6,7"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,7"),
            generate(f"{gen()}zero:6,3"),
        ],
        [
            generate(f"{gen()}zero:3,6"),
            generate(f"{gen()}zero:3,7"),
            generate(f"{gen()}diag:3"),
        ],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)

    rows = [
        [
            generate(f"{gen()}diag:6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
        ],
        [
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
            generate(f"{gen()}zero:6,6"),
        ],
    ]
    with AssertDenseWarning("could not preserve structure"):
        assert isinstance(B.block(*rows), Dense)
