"""ESA component blocks and satellite rows: conversion and strict validation.

Layout: [dx_1..dx_N, dy_1..dy_N, dz_1..dz_N,
         dvx_1..dvx_N, dvy_1..dvy_N, dvz_1..dvz_N].
Conversions retain numeric values without mutating inputs. Their default
permits synthetic layout examples, preserving the original decoder's behavior.
The evaluator always requests strict ESA decision bounds.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real


COMPONENTS = ("dx", "dy", "dz", "dvx", "dvy", "dvz")


def _satellite_count(n_sat):
    if isinstance(n_sat, bool) or not isinstance(n_sat, Integral) or n_sat <= 0:
        raise ValueError("Number of satellites must be a positive integer.")
    return int(n_sat)


def _sequence_length(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{label} must be an indexable numeric sequence.")
    if not hasattr(values, "__getitem__"):
        raise ValueError(f"{label} must be an indexable numeric sequence.")
    try:
        return len(values)
    except TypeError as error:
        raise ValueError(f"{label} must be an indexable numeric sequence.") from error


def chromosome_bounds(n_sat):
    """Return inclusive ESA bounds in official component-block order."""
    n_sat = _satellite_count(n_sat)
    return (
        [-1.0] * (3 * n_sat) + [-10.0] * (3 * n_sat),
        [1.0] * (3 * n_sat) + [10.0] * (3 * n_sat),
    )


def validate_chromosome(chromosome, n_sat, *, check_bounds=True):
    """Raise ValueError for invalid data; return None without copying inputs.

    Accept finite real scalars, including NumPy real scalars. Reject booleans,
    strings, complex values, nested vectors and nonfinite/overflowing values.
    Bounds describe initial decisions, not later satellite survival.
    """
    n_sat = _satellite_count(n_sat)
    if not isinstance(check_bounds, bool):
        raise ValueError("check_bounds must be a boolean.")
    if _sequence_length(chromosome, "Chromosome") != 6 * n_sat:
        raise ValueError(f"Invalid chromosome length: expected {6 * n_sat}.")
    for index, value in enumerate(chromosome):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Value at index {index} must be real, not boolean.")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"Value at index {index} is not representable.") from error
        if not finite:
            raise ValueError(f"Value at index {index} must be finite.")
        if check_bounds:
            limit = 1.0 if index < 3 * n_sat else 10.0
            if not -limit <= value <= limit:
                raise ValueError(f"Value at index {index} is outside [{-limit}, {limit}].")


def decode_chromosome(chromosome, n_sat, *, check_bounds=False):
    """Return N tuples (dx, dy, dz, dvx, dvy, dvz) from ESA blocks."""
    validate_chromosome(chromosome, n_sat, check_bounds=check_bounds)
    n_sat = int(n_sat)
    return [
        tuple(chromosome[component * n_sat + satellite] for component in range(6))
        for satellite in range(n_sat)
    ]


def encode_chromosome(satellites, n_sat=None, *, check_bounds=False):
    """Return ESA blocks from N rows of six values, inferring N if omitted.

    Values retain their numeric types; no float32 conversion or intermediate
    NumPy array is introduced. Inputs must support len() and indexing.
    """
    count = _sequence_length(satellites, "Satellites")
    n_sat = _satellite_count(count if n_sat is None else n_sat)
    if count != n_sat:
        raise ValueError(f"Expected {n_sat} satellite rows, received {count}.")
    for row in satellites:
        if _sequence_length(row, "Satellite row") != 6:
            raise ValueError("Each satellite row must contain exactly six components.")
    chromosome = [
        satellites[satellite][component]
        for component in range(6)
        for satellite in range(n_sat)
    ]
    validate_chromosome(chromosome, n_sat, check_bounds=check_bounds)
    return chromosome
