import lab as B

from matrix import Dense

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    dense1,
    dense2,
    diag1,
    diag2,
)


def test_concat(dense1, dense2, diag1, diag2):
    with AssertDenseWarning("concatenating <dense>, <dense>, <diagonal>..."):
        res = B.concat(dense1, dense2, diag1, diag2, axis=1)
        dense_res = B.concat(
            B.dense(dense1), B.dense(dense2), B.dense(diag1), B.dense(diag2), axis=1
        )
        approx(res, dense_res)
        assert isinstance(res, Dense)
