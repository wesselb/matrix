import lab as B
import pytest

from matrix import Diagonal, TiledBlocks, Zero

# noinspection PyUnresolvedReferences
from .util import AssertDenseWarning, approx, dense1


def test_tb_formatting():
    tb = TiledBlocks((Diagonal(B.ones(3)), 1), (Zero(int, 3, 3), 2), axis=0)
    assert str(tb) == "<tile of blocks: batch=(), shape=(9, 3), axis=0, dtype=float64>"
    assert repr(tb) == (
        "<tile of blocks: batch=(), shape=(9, 3), axis=0, dtype=float64\n"
        " <block: reps=1\n"
        "  matrix=<diagonal matrix: batch=(), shape=(3, 3), dtype=float64\n"
        "           diag=[1. 1. 1.]>\n"
        " <block: reps=2\n"
        "  matrix=<zero matrix: batch=(), shape=(3, 3), dtype=int>>"
    )


def test_tb_attributes():
    b1 = Diagonal(B.ones(3))
    b2 = Zero(int, 3, 3)

    tb = TiledBlocks((b1, 1), (b2, 2), axis=0)
    assert tb.blocks == (b1, b2)
    assert tb.reps == (1, 2)
    assert tb.axis == 0

    tb = TiledBlocks((b1, 1), (b2, 2), axis=1)
    assert tb.blocks == (b1, b2)
    assert tb.reps == (1, 2)
    assert tb.axis == 1


def test_tb_construction():
    b1 = Diagonal(B.ones(3))
    tb = TiledBlocks((b1, 2), axis=1)
    with AssertDenseWarning(["tiling", "concatenating"]):
        approx(tb, TiledBlocks(b1, b1, axis=1))
    with AssertDenseWarning(["tiling", "concatenating"]):
        approx(tb, TiledBlocks(b1, 2, axis=1))


def test_tb_checks():
    # Check that inputs must all be matrices.
    with pytest.raises(AssertionError):
        TiledBlocks(B.ones(3), Diagonal(B.ones(2)), axis=0)
