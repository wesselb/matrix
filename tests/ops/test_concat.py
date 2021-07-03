import lab as B

from matrix import Dense, TiledBlocks

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    dense1,
    dense2,
    diag1,
    diag2,
    tb1,
    tb2,
    tb_axis,
)


def test_concat(dense1, dense2, diag1, diag2):
    with AssertDenseWarning("concatenating <dense>, <dense>, <diagonal>..."):
        res = B.concat(dense1, dense2, diag1, diag2, axis=1)
        dense_res = B.concat(
            B.dense(dense1), B.dense(dense2), B.dense(diag1), B.dense(diag2), axis=1
        )
        approx(res, dense_res)
        assert isinstance(res, Dense)


def test_concat_tb(tb1, tb2):
    # Attempt to align them.
    if tb1.axis != tb2.axis:
        tb2 = B.transpose(tb2)

    if B.shape(tb1) == B.shape(tb2):
        # Test concatenating along tiling axis.
        res = B.concat(tb1, tb2, axis=tb1.axis)
        with AssertDenseWarning(["tiling", "concatenating"]):
            approx(B.concat(B.dense(tb1), B.dense(tb2), axis=tb1.axis), res)
        assert isinstance(res, TiledBlocks)

        # Test concatenating along other axis.
        with AssertDenseWarning("cannot nicely concatenate tiled blocks"):
            res = B.concat(tb1, tb2, axis=1 - tb1.axis)
        # No warnings here, because the dense versions are cached.
        approx(B.concat(B.dense(tb1), B.dense(tb2), axis=1 - tb1.axis), res)
        assert isinstance(res, Dense)
