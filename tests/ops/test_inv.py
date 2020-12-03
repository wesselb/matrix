import lab as B
import pytest

from matrix import Dense, Diagonal, Woodbury, Kronecker

# noinspection PyUnresolvedReferences
from ..util import approx, check_un_op, dense1, diag1, const1, lr1, wb1, kron1


def test_inv_dense(dense1):
    check_un_op(B.inv, dense1, asserted_type=Dense)


def test_inv_diag(diag1):
    check_un_op(B.inv, diag1, asserted_type=Diagonal)


def test_inv_wb(wb1):
    check_un_op(B.inv, wb1, asserted_type=Woodbury)


def test_inv_kron(kron1):
    check_un_op(B.inv, kron1, asserted_type=Kronecker)
