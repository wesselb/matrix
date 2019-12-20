import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense_pd
)


def test_logdet_dense(dense_pd):
    check_un_op(B.logdet, dense_pd)
