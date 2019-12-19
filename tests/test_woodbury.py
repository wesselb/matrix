import lab as B

from matrix import Diagonal, LowRank, Woodbury


def test_woodbury_str():
    diag = Diagonal(B.ones(3))
    lr = LowRank(B.ones(3, 1), 2 * B.ones(3, 1))
    assert str(Woodbury(diag, lr)) == \
           '<Woodbury matrix:\n' \
           ' diag=<diagonal matrix: shape=3x3, dtype=float64,\n' \
           '       diag=[1. 1. 1.]>,\n' \
           ' lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1,\n' \
           '     left=[[1.]\n' \
           '           [1.]\n' \
           '           [1.]],\n' \
           '     right=[[2.]\n' \
           '            [2.]\n' \
           '            [2.]]>>'
