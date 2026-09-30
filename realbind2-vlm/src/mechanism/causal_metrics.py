from __future__ import annotations

import random
from statistics import mean


def family_bootstrap(values_by_family: dict[str, list[float]], *, draws: int = 10000, seed: int = 42) -> dict:
    families = sorted(values_by_family)
    if not families:
        return {"mean": None, "ci95": [None, None], "draws": draws, "seed": seed}
    observed = mean(value for family in families for value in values_by_family[family])
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = [rng.choice(families) for _ in families]
        samples.append(mean(value for family in selected for value in values_by_family[family]))
    samples.sort()
    return {
        "mean": observed,
        "ci95": [samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]],
        "draws": draws,
        "seed": seed,
    }

