import lab as B

from matrix import Dense, Diagonal, Woodbury, Kronecker
# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    dense_pd,
    diag_pd,
    const_pd,
    lr_pd,
    wb_pd,
    kron_pd,
)


def test_pd_inv_correctness(dense_pd):
    approx(B.pd_inv(dense_pd), B.inv(dense_pd))


def test_pd_inv_dense(dense_pd):
    check_un_op(B.pd_inv, dense_pd, asserted_type=Dense)


def test_pd_inv_diag(diag_pd):
    check_un_op(B.pd_inv, diag_pd, asserted_type=Diagonal)


def test_pd_inv_wb(wb_pd):
    check_un_op(B.pd_inv, wb_pd, asserted_type=Woodbury)


def test_pd_inv_kron(kron_pd):
    check_un_op(B.pd_inv, kron_pd, asserted_type=Kronecker)
