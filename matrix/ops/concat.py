import lab as B
from wbml.warning import warn_upmodule

from ..matrix import AbstractMatrix, Dense, structured
from ..tiledblocks import TiledBlocks
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def concat(element: AbstractMatrix, *elements: AbstractMatrix, axis: B.Int = 0):
    elements = (element,) + elements

    if structured(*elements):
        elements_str = ", ".join(map(str, elements[:3]))
        if len(elements) > 3:
            elements_str += "..."
        warn_upmodule(
            f"Concatenating {elements_str}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.concat(*(B.dense(el) for el in elements), axis=axis))


@B.dispatch
def concat(element: TiledBlocks, *elements: TiledBlocks, axis: B.Int = 0):
    elements = (element,) + elements

    if not all([el.axis == axis for el in elements]):
        warn_upmodule(
            f"Cannot nicely concatenate tiled blocks: converting to dense.",
            category=ToDenseWarning,
        )
        return Dense(B.concat(*(B.dense(el) for el in elements), axis=axis))
    else:
        # All axes are aligned.
        blocks = sum([el.blocks for el in elements], ())
        reps = sum([el.reps for el in elements], ())
        return TiledBlocks(*zip(blocks, reps), axis=axis)
