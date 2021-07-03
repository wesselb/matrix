import lab as B
import pytest

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    dense_pd,
    diag_pd,
    kron_pd,
    lr_pd,
    lt_pd,
    ut_pd,
    wb_pd,
)


def test_logdet_dense(dense_pd):
    check_un_op(B.logdet, dense_pd)


def test_logdet_diag(diag_pd):
    check_un_op(B.logdet, diag_pd)


def test_logdet_lt(lt_pd):
    check_un_op(B.logdet, lt_pd)


def test_logdet_ut(ut_pd):
    check_un_op(B.logdet, ut_pd)


def test_logdet_wb(wb_pd):
    check_un_op(B.logdet, wb_pd)


def test_logdet_kron(kron_pd):
    check_un_op(B.logdet, kron_pd)
