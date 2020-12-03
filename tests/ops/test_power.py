import lab as B

from matrix import (
    Dense,
    Diagonal,
    Zero,
    Constant,
    LowerTriangular,
    UpperTriangular,
    Kronecker,
)

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    AssertDenseWarning,
    zero1,
    dense1,
    diag1,
    const1,
    lt1,
    ut1,
    lr1,
    wb1,
    kron1,
)


def power2(x):
    return B.power(x, 2)


def test_power_zero(zero1):
    check_un_op(power2, zero1, asserted_type=Zero)


def test_power_dense(dense1):
    check_un_op(power2, dense1, asserted_type=Dense)


def test_power_diag(diag1):
    check_un_op(power2, diag1, asserted_type=Diagonal)


def test_power_const(const1):
    check_un_op(power2, const1, asserted_type=Constant)


def test_power_lt(lt1):
    check_un_op(power2, lt1, asserted_type=LowerTriangular)


def test_power_ut(ut1):
    check_un_op(power2, ut1, asserted_type=UpperTriangular)


def test_power_lr(lr1):
    with AssertDenseWarning("power of <low-rank>"):
        check_un_op(power2, lr1, asserted_type=Dense)


def test_power_wb(wb1):
    with AssertDenseWarning("power of <woodbury>"):
        check_un_op(power2, wb1, asserted_type=Dense)


def test_power_kron(kron1):
    check_un_op(power2, kron1, asserted_type=Kronecker)
