import lab as B


def align(a, b):
    """Align two matrices according to identical columns.

    Args:
        a (matrix): First matrix to align.
        b (matrix): Second matrix to align.

    Returns:
        tuple[matrix]: A four tuple. The first two elements are permutations to
            *align* `a` and `b`. The second two elements are permutations to *join*
            `a` and `b`. *Important:* The permutations assume that the last column
            is a column of zeros.
    """
    if B.control_flow.use_cache:
        a_perm = B.control_flow.get_outcome("align:a_perm")
        b_perm = B.control_flow.get_outcome("align:b_perm")
        a_join_perm = B.control_flow.get_outcome("align:a_join_perm")
        b_join_perm = B.control_flow.get_outcome("align:b_join_perm")
        return a_perm, b_perm, a_join_perm, b_join_perm

    def equal(index_a, index_b):
        dist = B.mean(B.subtract(a[..., :, index_a], b[..., :, index_b]) ** 2)
        return dist < 1e-10

    # We need the norms later on.
    a_norms = B.sum(B.multiply(a, a), axis=0)
    b_norms = B.sum(B.multiply(b, b), axis=0)

    # Perform sorting to enable linear-time algorithm. These need to be Python lists
    # containing integers.
    a_sorted_inds = list(B.argsort(a_norms))
    b_sorted_inds = list(B.argsort(b_norms))

    a_perm = []
    b_perm = []
    a_join_perm = []
    b_join_perm = []

    while a_sorted_inds and b_sorted_inds:
        # Match at the first index.
        if equal(a_sorted_inds[0], b_sorted_inds[0]):
            a_ind = a_sorted_inds.pop(0)
            b_ind = b_sorted_inds.pop(0)
            a_perm.append(a_ind)
            b_perm.append(b_ind)
            a_join_perm.append(a_ind)
            b_join_perm.append(-1)
        # No match. Figure out which should be discarded.
        elif a_norms[a_sorted_inds[0]] < b_norms[b_sorted_inds[0]]:
            a_ind = a_sorted_inds.pop(0)
            a_perm.append(a_ind)
            b_perm.append(-1)
            a_join_perm.append(a_ind)
            b_join_perm.append(-1)
        else:
            b_ind = b_sorted_inds.pop(0)
            a_perm.append(-1)
            b_perm.append(b_ind)
            a_join_perm.append(-1)
            b_join_perm.append(b_ind)

    # Either `a_sorted_inds` or `b_sorted_inds` can have indices left.
    if a_sorted_inds:
        a_perm.extend(a_sorted_inds)
        b_perm.extend([-1] * len(a_sorted_inds))
        a_join_perm.extend(a_sorted_inds)
        b_join_perm.extend([-1] * len(a_sorted_inds))
    if b_sorted_inds:
        a_perm.extend([-1] * len(b_sorted_inds))
        b_perm.extend(b_sorted_inds)
        a_join_perm.extend([-1] * len(b_sorted_inds))
        b_join_perm.extend(b_sorted_inds)

    B.control_flow.set_outcome("align:a_perm", a_perm)
    B.control_flow.set_outcome("align:b_perm", b_perm)
    B.control_flow.set_outcome("align:a_join_perm", a_join_perm)
    B.control_flow.set_outcome("align:b_join_perm", b_join_perm)

    return a_perm, b_perm, a_join_perm, b_join_perm
