from __future__ import annotations

from fractions import Fraction
import math
import sys

SQRT_BIT_WIDTH = 2 * sys.float_info.mant_dig + 3


def _integer_sqrt_of_frac_rto(n: int, m: int) -> int:
    a = math.isqrt(n // m)
    return a | (a * a * m != n)


def float_sqrt_of_frac(n: int, m: int) -> float:
    q = (n.bit_length() - m.bit_length() - SQRT_BIT_WIDTH) // 2
    if q >= 0:
        numerator = _integer_sqrt_of_frac_rto(n, m << 2 * q) << q
        denominator = 1
    else:
        numerator = _integer_sqrt_of_frac_rto(n << -2 * q, m)
        denominator = 1 << -q
    return numerator / denominator


def sample_standard_deviation(values: list[float]) -> float:
    count = len(values)
    if count < 2:
        raise ValueError("standard deviation requires at least two values")
    exact = [Fraction(value) for value in values]
    total = sum(exact)
    total_squares = sum(value * value for value in exact)
    ss = (count * total_squares - total * total) / count
    mss = ss / (count - 1)
    return float_sqrt_of_frac(mss.numerator, mss.denominator)
