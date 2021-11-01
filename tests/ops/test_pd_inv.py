import lab as B

from matrix import Dense, Diagonal, Kronecker, Woodbury

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    const_pd,
    dense1_pd,
    diag1_pd,
    kron_pd,
    lr1_pd,
    wb1_pd,
)


def test_pd_inv_correctness(dense1_pd):
    approx(B.pd_inv(dense1_pd), B.inv(dense1_pd))


def test_pd_inv_dense(dense1_pd):
    check_un_op(B.pd_inv, dense1_pd, asserted_type=Dense)


def test_pd_inv_diag(diag1_pd):
    check_un_op(B.pd_inv, diag1_pd, asserted_type=Diagonal)


def test_pd_inv_wb(wb1_pd):
    check_un_op(B.pd_inv, wb1_pd, asserted_type=Woodbury)


def test_pd_inv_kron(kron_pd):
    check_un_op(B.pd_inv, kron_pd, asserted_type=Kronecker)
