"""Forecast GTA VI trophy earn-rate quantiles from Exophase history."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
WINDOW = 12
SHRINKAGE = 0.30
HORIZON_NOISE = 0.06


def clipped(rate):
    return min(99.99, max(0.01, float(rate)))


def logit(rate):
    rate = clipped(rate)
    return math.log(rate / (100.0 - rate))


def inv_logit(value):
    if value >= 0:
        e = math.exp(-value)
        return 100.0 / (1.0 + e)
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def round4(value):
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if result.is_integer() else result


def weighted_fit(xs, ys):
    """Return intercept, slope and weighted residual RMS, favoring recent reads."""
    count = len(xs)
    if count < 2 or max(ys) - min(ys) < 1e-12:
        return ys[-1], 0.0, 0.0
    weights = [0.65 ** (count - 1 - i) for i in range(count)]
    total = sum(weights)
    xbar = sum(w * x for w, x in zip(weights, xs)) / total
    ybar = sum(w * y for w, y in zip(weights, ys)) / total
    variance = sum(w * (x - xbar) ** 2 for w, x in zip(weights, xs))
    if variance < 1e-15:
        return ybar, 0.0, 0.0
    slope = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(weights, xs, ys)) / variance
    intercept = ybar - slope * xbar
    sigma = math.sqrt(sum(w * (y - intercept - slope * x) ** 2 for w, x, y in zip(weights, xs, ys)) / total)
    return intercept, slope, sigma


def strictly_increasing(values):
    """Round to four places while preserving five distinct valid percentages."""
    units = [min(999900, max(100, math.floor(v * 10000.0 + 0.5))) for v in values]
    for i in range(1, len(units)):
        units[i] = max(units[i], units[i - 1] + 1)
    if units[-1] > 999900:
        units[-1] = 999900
        for i in range(len(units) - 2, -1, -1):
            units[i] = min(units[i], units[i + 1] - 1)
    return [int(u / 10000) if u % 10000 == 0 else u / 10000 for u in units]


def forecast_case(case, name="case"):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for unsupported quantile levels")
    history = case.get("history", [])
    if not history:
        raise ValueError(f"{name} has no history")

    release = datetime.fromisoformat(case["game"]["release_date"]).replace(tzinfo=timezone.utc)
    first_live = next((i for i, read in enumerate(history) if timestamp(read["at"]) >= release), 0)
    reads = history[first_live:][-WINDOW:] or history[-WINDOW:]
    times = [timestamp(read["at"]) for read in reads]
    xs_all = [math.log1p(max(0.0, (at - release).total_seconds() / 60.0)) for at in times]
    target_x = math.log1p(max(0.0, (timestamp(case["target"]) - release).total_seconds() / 60.0))

    fits = []
    last_rates = []
    for index, _trophy in enumerate(case["trophies"]):
        points = [(x, logit(read["rates"][index])) for x, read in zip(xs_all, reads) if read["rates"][index] is not None]
        if not points:
            points = [(xs_all[-1], logit(0.01))]
        xs, ys = map(list, zip(*points))
        fits.append(weighted_fit(xs, ys))
        last_rates.append(inv_logit(ys[-1]))

    ordered = sorted(last_rates)
    cut1 = ordered[len(ordered) // 3]
    cut2 = ordered[(2 * len(ordered)) // 3]
    pools = {}
    keys = []
    for trophy, rate, fit in zip(case["trophies"], last_rates, fits):
        band = 0 if rate <= cut1 else 1 if rate <= cut2 else 2
        key = (trophy["tier"], band)
        keys.append(key)
        pools.setdefault(key, []).append(fit[1])
    pooled = {key: sum(slopes) / len(slopes) for key, slopes in pools.items()}

    tier_max = {}
    for trophy, rate in zip(case["trophies"], last_rates):
        tier_max[trophy["tier"]] = max(tier_max.get(trophy["tier"], -math.inf), logit(rate))
    tier_margin = {"platinum": 0.35, "gold": 0.50, "silver": 0.75, "bronze": 1.00}
    horizon_sigma = HORIZON_NOISE * math.sqrt(case["horizon_minutes"] / 60.0)

    rows = []
    for trophy, rate, fit, key in zip(case["trophies"], last_rates, fits, keys):
        intercept, slope, sigma = fit
        shrunk = (1.0 - SHRINKAGE) * slope + SHRINKAGE * pooled[key]
        prediction = intercept + shrunk * target_x
        last_logit = logit(rate)
        lower = last_logit - 0.20
        upper = max(last_logit + 0.05, tier_max[trophy["tier"]] + tier_margin.get(trophy["tier"], 0.75))
        prediction = min(upper, max(lower, prediction))
        sigma_eff = math.hypot(sigma, horizon_sigma)
        quantiles = [min(99.99, max(0.01, inv_logit(prediction + z * sigma_eff))) for z in Z]
        rows.append({"id": trophy["id"], "q": strictly_increasing(quantiles)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        output = forecast_case(case, path.name)
        destination = OUTPUT_DIR / f"{path.stem}.out"
        destination.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
