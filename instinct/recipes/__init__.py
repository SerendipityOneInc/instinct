"""Registry of released recipes, keyed by public model name."""

from .base import Encoded, Recipe
from .jqv_27b import RECIPE as JQV_27B
from .reflex_dual_4b import RECIPE as REFLEX_DUAL_4B
from .tuned_4b import RECIPE as TUNED_4B

RECIPES = {recipe.name: recipe for recipe in (JQV_27B, REFLEX_DUAL_4B, TUNED_4B)}


def get(name):
    try:
        return RECIPES[name]
    except KeyError:
        raise ValueError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}") from None


__all__ = ["Encoded", "Recipe", "RECIPES", "get"]
