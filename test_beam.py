import math
import random

import beam_python as bp


def test_direction_norm():
    random.seed(1)

    for i in range(1000):
        x, y, z = bp.random_direction()
        norm_squared = x**2 + y**2 + z**2

        assert abs(norm_squared - 1) < 1e-12

def test_direction_mean():
    random.seed(2)

    N = 1_000_000
    sx = 0.0
    sy = 0.0
    sz = 0.0

    for i in range(N):
        x, y, z = bp.random_direction()

        sx += x
        sy += y
        sz += z

    for total in (sx, sy, sz):
        assert abs(total / N) < 5 / math.sqrt(N)

       
def test_lorentz():
    random.seed(3)

    for i in range(10_000):
        beta = random.uniform(-0.99, 0.99)

        p4 = (
            random.uniform(0, 10),
            (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10)
            )
        )

        transformed = bp.lorentz_boost(beta, p4)
        restored = bp.lorentz_boost(-beta, transformed)

        assert math.isclose(
            bp.norm2(p4),
            bp.norm2(transformed),
            rel_tol=1e-9,
            abs_tol=1e-8
        )

        before = (p4[0], *p4[1])
        after = (restored[0], *restored[1])

        for x, y in zip(before, after):
            assert math.isclose(
                x, y,
                rel_tol=1e-9,
                abs_tol=1e-8
            )