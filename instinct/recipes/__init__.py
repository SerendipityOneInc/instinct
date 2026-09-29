"""Registry of recipes, keyed by model name: the released model and the reference baseline."""

from .base import Encoded, Recipe
from .jqv_27b import RECIPE as JQV_27B
from .tuned_4b import RECIPE as TUNED_4B

RECIPES = {recipe.name: recipe for recipe in (TUNED_4B, JQV_27B)}


def get(name):
    try:
        return RECIPES[name]
    except KeyError:
        raise ValueError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}") from None


__all__ = ["Encoded", "Recipe", "RECIPES", "get"]
