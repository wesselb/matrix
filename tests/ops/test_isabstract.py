import numpy as np
import jax.numpy as jnp
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
from ..util import dense1, diag1, const1, lt1, ut1, lr1, wb1, kron1


def check_isabstract(constructor, *args):
    # Regular application should not give anything abstract.
    assert not B.isabstract(constructor(*args))

    # Force each of the arguments to be abstract and check the result.
    for i in range(len(args)):
        tracked = []

        @B.jit
        def f(x):
            mat = constructor(*(x if j == i else args[j] for j in range(len(args))))
            tracked.append(B.isabstract(mat))
            return B.sum(mat)

        f(jnp.array(B.dense(args[i])))

        # First run should be concrete: populate control flow cache.
        assert not tracked[0]
        # Second run should be abstract.
        assert tracked[1]
        # And that should be all.
        assert len(tracked) == 2


def test_isabstract_zero():
    check_isabstract(lambda: Zero(np.float64, 2, 2))


def test_isabstract_dense(dense1):
    check_isabstract(Dense, dense1.mat)


def test_isabstract_diag(diag1):
    check_isabstract(Diagonal, diag1.diag)


def test_isabstract_const(const1):
    check_isabstract(
        lambda const: Constant(const, const1.rows, const1.cols), const1.const
    )


def test_isabstract_lt(lt1):
    check_isabstract(LowerTriangular, lt1.mat)


def test_isabstract_lt(ut1):
    check_isabstract(UpperTriangular, ut1.mat)


def test_isabstract_lr(lr1):
    def construct(left, right, middle):
        return LowRank(left, right, middle)

    check_isabstract(construct, lr1.left, lr1.right, lr1.middle)


def test_isabstract_wb(wb1):
    def construct(diag, left, right, middle):
        return Diagonal(diag) + LowRank(left, right, middle)

    check_isabstract(construct, wb1.diag.diag, wb1.lr.left, wb1.lr.right, wb1.lr.middle)


def test_isabstract_kron(kron1):
    check_isabstract(Kronecker, kron1.left, kron1.right)
