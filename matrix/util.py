from plum import Dispatcher
import lab as B

__all__ = ['indent',
           'dtype_str']

_dispatch = Dispatcher()


def indent(x, indentation='  '):
    """Indent a string.

    Args:
        x (str): String to indent.
        indentation (str, optional): Indentation. Defaults to two spaces.

    Returns:
        str: `x` indented.
    """
    return indentation + x.replace('\n', '\n' + indentation)


@_dispatch(object)
def dtype_str(x):
    """Get the data type of an object as string.

    Args:
        x (tensor): Tensor to get data type of.

    Returns:
        str: Data type of `x`.
    """
    return dtype_str(B.dtype(x))


@_dispatch(B.DType)
def dtype_str(dtype):
    if isinstance(dtype, type):
        return dtype.__name__
    else:
        return str(dtype)
