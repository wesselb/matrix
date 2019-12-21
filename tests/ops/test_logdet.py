import pytest
import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense_pd,
    wb_pd
)


def test_logdet_dense(dense_pd):
    check_un_op(B.logdet, dense_pd)


@pytest.mark.xfail
def test_logdet_wb(wb_pd):
    check_un_op(B.logdet, wb_pd)
