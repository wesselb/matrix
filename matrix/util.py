__all__ = ['indent']


def indent(x, indentation='  '):
    """Indent a string.

    Args:
        x (str): String to indent.
        indentation (str, optional): Indentation. Defaults to two spaces.

    Returns:
        str: `x` indented.
    """
    return indentation + x.replace('\n', '\n' + indentation)
