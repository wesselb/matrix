import lab as B
import numpy as np
import pytest

from matrix import Constant

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    dense_pd,
    diag_pd,
    kron_pd,
    lr_pd,
    wb_pd,
)


def _est_cov(a):
    num = 500_000
    samples = B.dense(B.sample(a, num=num))
    cov_est = B.matmul(samples, samples, tr_b=True) / num
    err = B.max(B.abs(B.dense(a) - cov_est)) / B.max(B.abs(B.dense(a)))
    assert err < 1e-1


def _check_state(a):
    state, sample1 = B.sample(B.create_random_state(B.dtype(a), seed=0), a)
    state, sample2 = B.sample(B.create_random_state(B.dtype(a), seed=0), a)
    assert isinstance(state, B.RandomState)
    approx(sample1, sample2)


def test_sample_conversion():
    const = Constant(1, 5, 5)
    sample = B.sample(const)
    assert B.issubdtype(B.dtype(sample), np.floating)


def test_sample_dense(dense_pd):
    _est_cov(dense_pd)
    _check_state(dense_pd)


def test_sample_diag(diag_pd):
    _est_cov(diag_pd)
    _check_state(diag_pd)


def test_sample_lr(lr_pd):
    _est_cov(lr_pd)
    _check_state(lr_pd)


def test_sample_wb(wb_pd):
    _est_cov(wb_pd)
    _check_state(wb_pd)


def test_sample_kron(kron_pd):
    _est_cov(kron_pd)
    _check_state(kron_pd)
