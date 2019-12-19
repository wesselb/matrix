from plum import Dispatcher, Self
import lab as B

from .diagonal import Diagonal
from .lowrank import LowRank
from .matrix import AbstractMatrix
from .shape import assert_compatible
from .util import indent, dtype_str

__all__ = ['Woodbury']


class Woodbury(AbstractMatrix):
    """Woodbury matrix.

    The data type of a Woodbury matrix is the data type of the low-rank part.

    Args:
        diag (:class:`.diagonal.Diagonal`): Diagonal part.
        lr (:class:`.diagonal.LowRank`): Low-rank part.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(Diagonal, LowRank)
    def __init__(self, diag, lr):
        assert_compatible(diag, lr)
        self.diag = diag
        self.lr = lr

    def __str__(self):
        rows, cols = B.shape(self)
        return f'<Woodbury matrix:\n' \
               f' shape={rows}x{cols},' \
               f' dtype={dtype_str(self)},\n' + \
               f' diag=' + indent(str(self.diag), ' ' * 6).strip() + ',\n' + \
               f' lr=' + indent(str(self.lr), ' ' * 4).strip() + '>'
