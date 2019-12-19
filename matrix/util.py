import lab as B

__all__ = ['indent',
           'dtype_str']


def indent(x, indentation='  '):
    """Indent a string.

    Args:
        x (str): String to indent.
        indentation (str, optional): Indentation. Defaults to two spaces.

    Returns:
        str: `x` indented.
    """
    return indentation + x.replace('\n', '\n' + indentation)


def dtype_str(x):
    """Get the data type of an object as string.

    Args:
        x (tensor): Tensor to get data type of.

    Returns:
        str: Data type of `x`.
    """
    dtype = B.dtype(x)
    if isinstance(dtype, type):
        return dtype.__name__
    else:
        return str(dtype)
