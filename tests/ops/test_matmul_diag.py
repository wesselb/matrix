import lab as B

from matrix import structured

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    AssertDenseWarning,
    ConditionalContext,

    dense1,
    dense2,
    lr1,
    lr2
)


def _check_matmul_diag(a, b):
    for tr_a in [False, True]:
        for tr_b in [False, True]:
            allclose(B.matmul_diag(a, b, tr_a=tr_a, tr_b=tr_b),
                     B.diag(B.matmul(B.dense(a), B.dense(b),
                                     tr_a=tr_a, tr_b=tr_b)))


def test_matmul_diag_dense(dense1, dense2):
    _check_matmul_diag(dense1, dense2)


def test_matmul_diag_lr(lr1, lr2):
    warn = AssertDenseWarning('getting the diagonal of <low-rank>')
    with ConditionalContext(structured(lr1.left, lr1.right) and
                            structured(lr2.left, lr2.right),
                            warn):
        _check_matmul_diag(lr1, lr2)
