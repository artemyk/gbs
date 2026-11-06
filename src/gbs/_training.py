"""Validation and learning-rate schedules shared by the fitting stages."""
import math
from numbers import Integral, Real


def require_integer(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_finite(name, value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


def cosine_fraction(step, steps, minimum=.1):
    """Fraction of the initial learning rate; evaluated before each update."""
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*step/steps))
