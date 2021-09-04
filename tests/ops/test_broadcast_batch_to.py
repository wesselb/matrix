import lab as B

from matrix import (
    Zero,
    Dense,
    Diagonal,
    Constant,
    LowerTriangular,
    UpperTriangular,
    LowRank,
    Woodbury,
    Kronecker,
)

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    check_un_op,
    zero1,
    dense_bc,
    diag1,
    const1,
    lt1,
    ut1,
    lr1,
    wb1,
    kron1,
)


def _broadcast_batch(x):
    return B.broadcast_batch_to(x, 3, 3)


def test_broadcast_batch_to_zero(zero1):
    check_un_op(_broadcast_batch, zero1, asserted_type=Zero)


def test_broadcast_batch_to_dense(dense_bc):
    check_un_op(_broadcast_batch, dense_bc, asserted_type=Dense)


def test_broadcast_batch_to_diag(diag1):
    check_un_op(_broadcast_batch, diag1, asserted_type=Diagonal)


def test_broadcast_batch_to_const(const1):
    check_un_op(_broadcast_batch, const1, asserted_type=Constant)


def test_broadcast_batch_to_lt(lt1):
    check_un_op(_broadcast_batch, lt1, asserted_type=LowerTriangular)


def test_broadcast_batch_to_ut(ut1):
    check_un_op(_broadcast_batch, ut1, asserted_type=UpperTriangular)


def test_broadcast_batch_to_lr(lr1):
    check_un_op(_broadcast_batch, lr1, asserted_type=LowRank)


def test_broadcast_batch_to_wb(wb1):
    check_un_op(_broadcast_batch, wb1, asserted_type=Woodbury)


def test_broadcast_batch_to_kron(kron1):
    check_un_op(_broadcast_batch, kron1, asserted_type=Kronecker)
