import numpy as np
import lab as B


def align(a, with_a, b, with_b):
    """Align two matrices according to identical columns.

    Args:
        a (matrix): First matrix to align.
        with_a (matrix): Another matrix that will be aligned according to the columns
            of `a`.
        b (matrix): Second matrix to align.
        with_b (matrix): Another matrix that will be aligned according to the columns
            of `b`.

    Returns:
        tuple[matrix]: The union of the columns of `a` and `b`; `a` with columns
            aligned with `b` and zeros inserted where there is no match; `with_a` in
            the same ordering; `b` with columns aligned with `a` and zeros inserted
            where there is no match; and `with_b` in the same ordering.
    """
    # Sort the columns of `a` and `b` according to a norm so that a greedy approach
    # is possible. Also cache control flow, which will allow us to JIT.
    a_norms = B.sum(B.multiply(a, a), axis=0)
    b_norms = B.sum(B.multiply(b, b), axis=0)
    if B.control_flow.caching:
        B.control_flow.set_outcome("align:a_norms", a_norms)
        B.control_flow.set_outcome("align:b_norms", b_norms)
    elif B.control_flow.use_cache:
        a_norms = B.control_flow.get_outcome("align:a_norms")
        b_norms = B.control_flow.get_outcome("align:b_norms")
    a_inds = B.argsort(a_norms)
    b_inds = B.argsort(b_norms)
    join, a, with_a, b, with_b, n_extended = _align(
        a[:, a_inds],
        with_a[:, a_inds],
        a_norms[a_inds],
        b[:, b_inds],
        with_b[:, b_inds],
        b_norms[b_inds],
        0,
    )

    # Remove the added zeros. Cache this number of zeros to allow us to JIT.
    if B.control_flow.caching:
        B.control_flow.set_outcome("align:n_extended", n_extended, type=int)
    elif B.control_flow.use_cache:
        n_extended = B.control_flow.get_outcome("align:n_extended")
    return (
        join[:, 0 : B.shape(join)[1] - n_extended],
        a[:, 0 : B.shape(a)[1] - n_extended],
        with_a[:, 0 : B.shape(with_a)[1] - n_extended],
        b[:, 0 : B.shape(b)[1] - n_extended],
        with_b[:, 0 : B.shape(with_b)[1] - n_extended],
    )


def _n_extend(x, n):
    return B.concat(x, B.zeros(B.dtype(x), B.shape(x)[0], n), axis=1)


def _align(a, with_a, a_norms, b, with_b, b_norms, n_extend=0):
    # Check base cases.
    if B.shape(a)[1] == 0:
        excess = B.shape(b)[1]
        return (
            _n_extend(b, n_extend),
            _n_extend(B.zeros(B.dtype(a), B.shape(a)[0], excess), n_extend),
            _n_extend(B.zeros(B.dtype(with_a), B.shape(with_a)[0], excess), n_extend),
            _n_extend(b, n_extend),
            _n_extend(with_b, n_extend),
            n_extend,
        )
    if B.shape(b)[1] == 0:
        excess = B.shape(a)[1]
        return (
            _n_extend(a, n_extend),
            _n_extend(a, n_extend),
            _n_extend(with_a, n_extend),
            _n_extend(B.zeros(B.dtype(b), B.shape(b)[0], excess), n_extend),
            _n_extend(B.zeros(B.dtype(with_b), B.shape(with_b)[0], excess), n_extend),
            n_extend,
        )
    # Not in the base cases. Recurse.
    return B.cond(
        B.sum(B.subtract(a[:, 0], b[:, 0]) ** 2) < 1e-10,
        lambda *xs: _match(*xs, n_extend),
        lambda *xs: _mismatch(*xs, n_extend),
        a,
        with_a,
        a_norms,
        b,
        with_b,
        b_norms,
    )


def _match(a, with_a, a_norms, b, with_b, b_norms, n_extend):
    join_rest, a_rest, with_a_rest, b_rest, with_b_rest, n_extended = _align(
        a[:, 1:],
        with_a[:, 1:],
        a_norms[1:],
        b[:, 1:],
        with_b[:, 1:],
        b_norms[1:],
        n_extend + 1,
    )
    return (
        B.concat(a[:, :1], join_rest, axis=1),  # Can use either.
        B.concat(a[:, :1], a_rest, axis=1),
        B.concat(with_a[:, :1], with_a_rest, axis=1),
        B.concat(b[:, :1], b_rest, axis=1),
        B.concat(with_b[:, :1], with_b_rest, axis=1),
        n_extended,
    )


def _mismatch(a, with_a, a_norms, b, with_b, b_norms, n_extend):
    return B.cond(
        a_norms[0] < b_norms[0],
        lambda *xs: _pick_a(*xs, n_extend),
        lambda *xs: _pick_b(*xs, n_extend),
        a,
        with_a,
        a_norms,
        b,
        with_b,
        b_norms,
    )


def _pick_a(a, with_a, a_norms, b, with_b, b_norms, n_extend):
    join_rest, a_rest, with_a_rest, b, with_b, n_extended = _align(
        a[:, 1:], with_a[:, 1:], a_norms[1:], b, with_b, b_norms, n_extend
    )
    return (
        B.concat(a[:, :1], join_rest, axis=1),
        B.concat(a[:, :1], a_rest, axis=1),
        B.concat(with_a[:, :1], with_a_rest, axis=1),
        B.concat(B.zeros(B.dtype(b), B.shape(b)[0], 1), b, axis=1),
        B.concat(B.zeros(B.dtype(with_b), B.shape(with_b)[0], 1), with_b, axis=1),
        n_extended,
    )


def _pick_b(a, with_a, a_norms, b, with_b, b_norms, n_extend):
    join_rest, a, with_a, b_rest, with_b_rest, n_extended = _align(
        a, with_a, a_norms, b[:, 1:], with_b[:, 1:], b_norms[1:], n_extend
    )
    return (
        B.concat(b[:, :1], join_rest, axis=1),
        B.concat(B.zeros(B.dtype(a), B.shape(a)[0], 1), a, axis=1),
        B.concat(B.zeros(B.dtype(with_a), B.shape(with_a)[0], 1), with_a, axis=1),
        B.concat(b[:, :1], b_rest, axis=1),
        B.concat(with_b[:, :1], with_b_rest, axis=1),
        n_extended,
    )
