import re
import warnings
from itertools import product

import lab as B
import numpy as np
import pytest
from numpy.testing import assert_allclose
from plum import Dispatcher, Union

from matrix import (
    Constant,
    Dense,
    Diagonal,
    Kronecker,
    LowerTriangular,
    LowRank,
    TiledBlocks,
    UpperTriangular,
    Woodbury,
    Zero,
)
from matrix.ops.util import align_batch
from matrix.util import ToDenseWarning

__all__ = [
    "approx",
    "check_un_op",
    "check_bin_op",
    "IgnoreDenseWarning",
    "AssertDenseWarning",
    "ConditionalContext",
    "concat_warnings",
    # Fixtures:
    "gen_batch_code",
    "mat1",
    "mat2",
    "vec1",
    "vec2",
    "scalar1",
    "scalar2",
    "zero1",
    "zero2",
    "zero_r",
    "dense1",
    "dense2",
    "dense_bc",
    "dense_r",
    "dense1_pd",
    "dense2_pd",
    "diag1",
    "diag2",
    "diag1_pd",
    "diag2_pd",
    "const1",
    "const2",
    "const_r",
    "const_pd",
    "const_or_scalar1",
    "const_or_scalar2",
    "lt1",
    "lt2",
    "lt_pd",
    "ut1",
    "ut2",
    "ut_pd",
    "lr1",
    "lr2",
    "lr_r",
    "lr1_pd",
    "lr2_pd",
    "wb1",
    "wb2",
    "wb1_pd",
    "wb2_pd",
    "kron1",
    "kron2",
    "kron_r",
    "kron_pd",
    "kron_mixed",
    "tb_axis",
    "tb1",
    "tb2",
]

_dispatch = Dispatcher()


@_dispatch
def approx(x, y, rtol=1e-7, atol=1e-12):
    """Assert that two objects are numerically close.

    Args:
        x (object): First object.
        y (object): Second object.
        rtol (float, optional): Relative tolerance. Defaults to `1e-7`.
        atol (float, optional): Absolute tolerance. Defaults to `1e-12`.
    """
    approx(B.to_numpy(x), B.to_numpy(y), rtol=rtol, atol=atol)


@_dispatch
def approx(
    x: Union[tuple, B.Number, B.NPNumeric],
    y: Union[tuple, B.Number, B.NPNumeric],
    **kw_args,
):
    assert_allclose(x, y, **kw_args)
    assert B.shape(x) == B.shape(y)


def _assert_instance(x, asserted_type):
    assert isinstance(
        x, asserted_type
    ), f"Expected instance of type {asserted_type} but got {type(x)}."


def check_un_op(op, x, ref=None, asserted_type=object):
    """Assert the correct of a unary operation by checking whether the
    result is the same on the dense version of the argument.

    Args:
        op (function): Unary operation to check.
        ref (function, optional): Set this as the function to be used as the reference
            to which the dense matrices will be fed.
        x (object): Argument.
        asserted_type (type, optional): Type of result.
    """
    if ref:
        op_ref = ref
    else:
        op_ref = op

    x_dense = B.dense(x)

    res = op(x)
    res_ref = op_ref(x_dense)
    approx(res, res_ref)

    _assert_instance(res, asserted_type)


def check_bin_op(
    op,
    x,
    y,
    ref=None,
    asserted_type=object,
    check_broadcasting=True,
    align_dense_batch=False,
):
    """Assert the correct of a binary operation by checking whether the
    result is the same on the dense versions of the arguments.

    Args:
        op (function): Binary operation to check.
        x (object): First argument.
        y (object): Second argument.
        ref (function, optional): Set this as the function to be used as the reference
            to which the dense matrices will be fed.
        asserted_type (type, optional): Type of result.
        check_broadcasting (bool, optional): Check broadcasting behaviour.
        align_dense_batch (bool, optional): Broadcast the batch dimensions of the
            dense matrices to check against. This can be used if the operation for
            dense matrices does not support broadcasting of batch dimensions.
    """
    if ref:
        op_ref = ref
    else:
        op_ref = op

    x_dense = B.dense(x)
    y_dense = B.dense(y)
    if align_dense_batch:
        x_dense, y_dense = align_batch(x_dense, y_dense)

    res = op(x, y)
    res_ref = op_ref(x_dense, y_dense)
    approx(res, res_ref)

    # Test any possible caching.
    approx(op(x, y), op_ref(x_dense, y_dense))

    with IgnoreDenseWarning():
        if check_broadcasting:
            approx(op(x_dense, y), res_ref)
            approx(op(x, y_dense), res_ref)

    _assert_instance(res, asserted_type)


def _sanitise(msg):
    # Filter details from printed objects
    msg = re.sub(r"<([a-zA-Z\- ]{1,})( matrix| product):[^>]*>", r"<\1>", msg)
    return msg


class IgnoreDenseWarning:
    """Context for ignoring `ToDenseWarnings`."""

    def __enter__(self):
        warnings.filterwarnings(category=ToDenseWarning, action="ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filterwarnings(category=ToDenseWarning, action="default")


class AssertDenseWarning:
    """Assert that a `ToDenseWarning` is raised with a particular content.

    Args:
        content (str or list[str]): Content that the arguments of the warning
            must contain.
    """

    def __init__(self, content):
        if isinstance(content, str):
            self.content = [content.lower()]
        else:
            self.content = [x.lower() for x in content]
        self.record = None

    def __enter__(self):
        self.context = pytest.warns(ToDenseWarning)
        self.record = self.context.__enter__()
        return self.record

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.__exit__(exc_type, exc_val, exc_tb)

        # Perform assertions.
        for i in range(len(self.record)):
            message = "".join(self.record[i].message.args)
            message = _sanitise(message).lower()

            if not any(content in message for content in self.content):
                pre = "any of " if len(self.content) > 1 else ""
                contents = '", "'.join(self.content)
                raise AssertionError(
                    f'Warning should contain {pre}"{contents}", but it did not: '
                    f'"{message}".'
                )


class ConditionalContext:
    """A conditional context mananger.

    Args:
        condition (bool): Boolean indicating whether `context` should be
            entered.
        context (object): Context to enter in the case that `condition` is
            `True`.
    """

    def __init__(self, condition, context):
        self.condition = condition
        self.context = context

    def __enter__(self):
        if self.condition:
            self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition:
            self.context.__exit__(exc_type, exc_val, exc_tb)


def generate(code):
    """Generate a random tensor of a particular type, specified with a code.

    Args:
        code (str): Code of the matrix.

    Returns:
        tensor: Random tensor.
    """
    # Handle batch shape:
    batch_code, code = code.split("|")

    # Parse batch.
    if batch_code == "":
        batch = ()
    else:
        batch = tuple(int(d) for d in batch_code.split(","))

    # Handle matrix shape:
    mat_code, shape_code = code.split(":")

    # Parse shape.
    if shape_code == "":
        shape = ()
    else:
        shape = tuple(int(d) for d in shape_code.split(","))

    if mat_code == "randn":
        return B.randn(*batch, *shape)
    elif mat_code == "randn_pd":
        mat = B.randn(*batch, *shape)

        # If it is a scalar or vector, just pointwise square it.
        if len(shape) in {0, 1}:
            return mat ** 2 + 1
        else:
            return B.matmul(mat, mat, tr_b=True) + B.eye(*batch, shape[0], shape[0])

    elif mat_code == "zero":
        return Zero(B.default_dtype, *batch, *shape)

    elif mat_code == "const":
        return Constant(B.randn(*batch), *shape)
    elif mat_code == "const_pd":
        return Constant(B.randn(*batch) ** 2 + 1, *shape)

    elif mat_code == "lt":
        mat = B.vec_to_tril(B.randn(*batch, int(0.5 * shape[0] * (shape[0] + 1))))
        return LowerTriangular(mat)
    elif mat_code == "lt_pd":
        mat = generate(f"{batch_code}|randn_pd:{shape[0]},{shape[0]}")
        return LowerTriangular(B.cholesky(mat))

    elif mat_code == "ut":
        mat = B.vec_to_tril(B.randn(*batch, int(0.5 * shape[0] * (shape[0] + 1))))
        return UpperTriangular(B.transpose(mat))
    elif mat_code == "ut_pd":
        mat = generate(f"{batch_code}|randn_pd:{shape[0]},{shape[0]}")
        return UpperTriangular(B.transpose(B.cholesky(mat)))

    elif mat_code == "dense":
        return Dense(generate(f"{batch_code}|randn:{shape_code}"))
    elif mat_code == "dense_pd":
        return Dense(generate(f"{batch_code}|randn_pd:{shape_code}"))

    elif mat_code == "diag":
        return Diagonal(generate(f"{batch_code}|randn:{shape_code}"))
    elif mat_code == "diag_pd":
        return Diagonal(generate(f"{batch_code}|randn_pd:{shape_code}"))

    else:
        raise RuntimeError(f'Cannot parse generation code "{code}".')


#: Warnings for concatenations.
concat_warnings = [
    "concatenating <dense>, <diagonal>",
    "concatenating <diagonal>, <dense>",
    "concatenating <diagonal>, <diagonal>",
    "constructing a dense block-diagonal matrix from <dense> and <diagonal>",
    "constructing a dense block-diagonal matrix from <diagonal> and <dense>",
    "constructing a dense block-diagonal matrix from <dense> and <dense>",
]


# Fixtures:

batch_prefixes = ["|", "3|"]


@pytest.fixture(params=[0, 1, 2])
def gen_batch_code(request):
    if request.param == 0:
        # No batches
        return lambda: "|"
    elif request.param == 1:
        # All batched
        return lambda: "3|"
    else:
        # Random mixture
        return lambda: np.random.choice(batch_prefixes)


@_dispatch
def loop_batches(code: str):
    return [prefix + code for prefix in batch_prefixes]


@_dispatch
def loop_batches(codes: tuple):
    return list(product(*(loop_batches(code) for code in codes)))


@_dispatch
def loop_batches(codes: list):
    return [prefixed_code for code in codes for prefixed_code in loop_batches(code)]


@pytest.fixture(params=loop_batches("randn:6,6"))
def mat1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("randn:6,6"))
def mat2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("randn:6"))
def vec1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("randn:6"))
def vec2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("randn:"))
def scalar1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("randn:"))
def scalar2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("zero:6,6"))
def zero1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("zero:6,6"))
def zero2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("zero:6,4"))
def zero_r(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("dense:6,6"))
def dense1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("dense:6,6"))
def dense2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches(["dense:6,1", "dense:1,6", "dense:6,6"]))
def dense_bc(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("dense:6,4"))
def dense_r(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("dense_pd:6,6"))
def dense1_pd(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("dense_pd:6,6"))
def dense2_pd(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("diag:6"))
def diag1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("diag:6"))
def diag2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("diag_pd:6"))
def diag1_pd(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("diag_pd:6"))
def diag2_pd(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("const:6,6"))
def const1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("const:6,6"))
def const2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("const:6,4"))
def const_r(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("const_pd:6,6"))
def const_pd(request):
    return generate(request.param)


@pytest.fixture(params=["|randn:"] + loop_batches(["const:6,6"]))
def const_or_scalar1(request):
    return generate(request.param)


@pytest.fixture(params=["|randn:"] + loop_batches(["const:6,6"]))
def const_or_scalar2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("lt:6"))
def lt1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("lt:6"))
def lt2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("lt_pd:6"))
def lt_pd(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("ut:6"))
def ut1(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("ut:6"))
def ut2(request):
    return generate(request.param)


@pytest.fixture(params=loop_batches("ut_pd:6"))
def ut_pd(request):
    return generate(request.param)


@pytest.fixture(
    params=loop_batches(
        [
            ("dense:6,1", "dense:1,1"),
            ("dense:6,2", "dense:2,2"),
            ("dense:6,3", "dense:3,3"),
            ("diag:6", "dense:6,6"),
        ]
    )
)
def lr1(request):
    code_lr, code_m = request.param
    return LowRank(generate(code_lr), generate(code_lr), middle=generate(code_m))


@pytest.fixture(
    params=loop_batches(
        [
            ("dense:6,1", "dense:1,1"),
            ("dense:6,2", "dense:2,2"),
            ("dense:6,3", "dense:3,3"),
            ("diag:6", "dense:6,6"),
        ]
    )
)
def lr2(request):
    code_lr, code_m = request.param
    return LowRank(generate(code_lr), generate(code_lr), middle=generate(code_m))


@pytest.fixture(
    params=loop_batches(
        [
            ("dense:6,1", "dense:4,1", "dense:1,1"),
            ("dense:6,2", "dense:4,2", "dense:2,2"),
            ("dense:6,3", "dense:4,3", "dense:3,3"),
            ("diag:6", "dense:4,6", "dense:6,6"),
        ]
    )
)
def lr_r(request):
    code_l, code_r, code_m = request.param
    return LowRank(generate(code_l), generate(code_r), middle=generate(code_m))


@pytest.fixture(
    params=loop_batches(
        [
            ("dense:6,1", "dense_pd:1,2"),
            ("dense:6,2", "dense_pd:2,2"),
            ("dense:6,3", "dense_pd:3,3"),
            ("diag:6", "diag_pd:6"),
        ]
    )
)
def lr1_pd(request):
    code_l, code_m = request.param
    return LowRank(generate(code_l), middle=generate(code_m))


@pytest.fixture(
    params=loop_batches(
        [
            ("dense:6,1", "dense_pd:1,2"),
            ("dense:6,2", "dense_pd:2,2"),
            ("dense:6,3", "dense_pd:3,3"),
            ("diag:6", "diag_pd:6"),
        ]
    )
)
def lr2_pd(request):
    code_l, code_m = request.param
    return LowRank(generate(code_l), middle=generate(code_m))


@pytest.fixture()
def wb1(diag1, lr1):
    return Woodbury(diag1, lr1)


@pytest.fixture()
def wb2(diag2, lr2):
    return Woodbury(diag2, lr2)


@pytest.fixture()
def wb1_pd(diag1_pd, lr1_pd):
    return Woodbury(diag1_pd, lr1_pd)


@pytest.fixture()
def wb2_pd(diag2_pd, lr2_pd):
    return Woodbury(diag2_pd, lr2_pd)


@pytest.fixture(
    params=loop_batches(
        [
            ("diag:1", "diag:6"),
            ("diag:2", "diag:3"),
            ("diag:3", "diag:2"),
            ("diag:6", "diag:1"),
            ("dense:1,1", "dense:6,6"),
            ("dense:2,2", "dense:3,3"),
            ("dense:3,3", "dense:2,2"),
            ("dense:6,6", "dense:1,1"),
        ]
    )
)
def kron1(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(
    params=loop_batches(
        [
            ("diag:1", "diag:6"),
            ("diag:2", "diag:3"),
            ("diag:3", "diag:2"),
            ("diag:6", "diag:1"),
            ("dense:1,1", "dense:1,1"),
            ("dense:2,2", "dense:3,3"),
            ("dense:3,3", "dense:2,2"),
            ("dense:6,6", "dense:1,1"),
        ]
    )
)
def kron2(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(
    params=loop_batches(
        [
            ("diag:2", "dense:3,2"),
            ("dense:3,2", "diag:2"),
            ("dense:1,4", "dense:6,1"),
            ("dense:6,1", "dense:1,4"),
        ]
    )
)
def kron_r(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(
    params=loop_batches(
        [
            ("diag_pd:1", "diag_pd:6"),
            ("diag_pd:2", "diag_pd:3"),
            ("diag_pd:3", "diag_pd:2"),
            ("diag_pd:6", "diag_pd:1"),
            ("dense_pd:1,1", "dense_pd:6,6"),
            ("dense_pd:2,2", "dense_pd:3,3"),
            ("dense_pd:3,3", "dense_pd:2,2"),
            ("dense_pd:6,6", "dense_pd:1,1"),
        ]
    )
)
def kron_pd(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(
    params=loop_batches(
        [
            ("diag:1", "diag:6"),
            ("diag:2", "diag:3"),
            ("diag:3", "diag:2"),
            ("diag:6", "diag:1"),
            ("dense:1,6", "dense:6,1"),
            ("dense:2,3", "dense:3,2"),
            ("dense:3,2", "dense:2,3"),
            ("dense:6,1", "dense:1,6"),
        ]
    )
)
def kron_mixed(request):
    code1, code2 = request.param
    return Kronecker(generate(code1), generate(code2))


@pytest.fixture(params=[-2, -1, 0, 1, 2])
def tb_axis(request):
    return request.param


_tb_codes = [
    [(c1, 2), (c2, 3), (c3, 4)]
    for c1, c2, c3 in loop_batches([("dense:3,3", "zero:3,3", "lt:3,3")])
] + [
    [(c1, 2), (c2, 3), (c3, 1)]
    for c1, c2, c3 in loop_batches([("diag:3", "ut:3,3", "diag:3")])
]


@pytest.fixture(params=_tb_codes)
def tb1(request, tb_axis):
    # Ensure that the axis is compatible.
    if tb_axis in {-2, 2} and all(code.startswith("|") for code, _ in request.param):
        if tb_axis == -2:
            tb_axis = -1
        else:
            tb_axis = 1
    return TiledBlocks(
        *((generate(code), rep) for code, rep in request.param), axis=tb_axis
    )


@pytest.fixture(params=_tb_codes)
def tb2(request, tb_axis):
    # Ensure that the axis is compatible.
    if tb_axis in {-2, 2} and all(code.startswith("|") for code, _ in request.param):
        if tb_axis == -2:
            tb_axis = -1
        else:
            tb_axis = 1
    return TiledBlocks(
        *((generate(code), rep) for code, rep in request.param), axis=tb_axis
    )
