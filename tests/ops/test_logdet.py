import lab as B
import pytest

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    dense1_pd,
    diag1_pd,
    kron_pd,
    lr1_pd,
    lt_pd,
    ut_pd,
    wb1_pd,
)


def test_logdet_dense(dense1_pd):
    check_un_op(B.logdet, dense1_pd)


def test_logdet_diag(diag1_pd):
    check_un_op(B.logdet, diag1_pd)


def test_logdet_lt(lt_pd):
    check_un_op(B.logdet, lt_pd)


def test_logdet_ut(ut_pd):
    check_un_op(B.logdet, ut_pd)


def test_logdet_wb(wb1_pd):
    check_un_op(B.logdet, wb1_pd)


def test_logdet_kron(kron_pd):
    check_un_op(B.logdet, kron_pd)
