from plum import Dispatcher
import lab as B

__all__ = ["ToDenseWarning", "indent", "dtype_str", "redirect"]

_dispatch = Dispatcher()


class ToDenseWarning(UserWarning):
    """Warnings that are generated when matrices are converted to dense."""


def indent(x, indentation="  "):
    """Indent a string.

    Args:
        x (str): String to indent.
        indentation (str, optional): Indentation. Defaults to two spaces.

    Returns:
        str: `x` indented.
    """
    return indentation + x.replace("\n", "\n" + indentation)


@_dispatch
def dtype_str(x):
    """Get the data type of an object as string.

    Args:
        x (tensor): Tensor to get data type of.

    Returns:
        str: Data type of `x`.
    """
    return dtype_str(B.dtype(x))


@_dispatch
def dtype_str(dtype: B.DType):
    if isinstance(dtype, type):
        return dtype.__name__
    else:
        return str(dtype)


def redirect(f, types_from, types_to, reverse=True):
    """Redirect a call with particular types to a method with other types.

    Args:
        f (:class:`.plum.Function`): Function.
        types_from (tuple[type]): Types of the method to redirect from.
        types_to (tuple[type]): Types of the method to redirect to.
        reverse (bool, optional): Perform the same redirection for the types
            reversed. Defaults to `True`.
    """
    target_method = f.invoke(*types_to)
    f.dispatch_multi(types_from)(target_method)
    if reverse and len(types_from) > 1:
        target_method = f.invoke(*reversed(types_to))
        f.dispatch_multi(tuple(reversed(types_from)))(target_method)