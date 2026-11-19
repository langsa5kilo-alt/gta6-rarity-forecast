"""Adaptive logit forecasts for GTA VI trophy earn rates."""

import json
import math
import os
from datetime import datetime
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
HISTORY_WINDOW = 12
BASE_SHRINKAGE = 0.30

# Practice one-step backtest (cuts 3 and 4, averaged across p00 and p01):
# starter persistence  0.030090 / 0.030090 / mean 0.030090
# adaptive model       0.005032 / 0.005032 / mean 0.005032


def clamp_rate(rate):
    return min(99.99, max(0.01, float(rate)))


def logit(rate):
    rate = clamp_rate(rate)
    return math.log(rate / (100.0 - rate))


def inv_logit(value):
    if value >= 0.0:
        decay = math.exp(-value)
        return 100.0 / (1.0 + decay)
    growth = math.exp(value)
    return 100.0 * growth / (1.0 + growth)


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def linear_fit(xs, ys):
    """Fit a line and return intercept, slope, residual SD, and R squared."""
    count = len(xs)
    if count < 2:
        return ys[-1], 0.0, 0.0, 0.0
    x_mean = sum(xs) / count
    y_mean = sum(ys) / count
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    if x_ss < 1e-15:
        return y_mean, 0.0, 0.0, 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / x_ss
    intercept = y_mean - slope * x_mean
    residual_ss = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    residual_sd = math.sqrt(residual_ss / (count - 2)) if count > 2 else 0.0
    total_ss = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0
    return intercept, slope, residual_sd, min(1.0, max(0.0, r_squared))


def evidence_weighted_slope(slope, pooled_slope, sample_count, r_squared):
    """Shrink a trophy slope toward its pool less strongly as evidence rises."""
    evidence = max(0, sample_count - 2) * max(0.0, r_squared)
    pool_weight = BASE_SHRINKAGE / (1.0 + evidence)
    return (1.0 - pool_weight) * slope + pool_weight * pooled_slope


def valid_quantiles(values):
    """Round half up to four places and keep adjacent quantiles distinct."""
    units = [min(999900, max(100, math.floor(value * 10000.0 + 0.5))) for value in values]
    for index in range(1, len(units)):
        units[index] = max(units[index], units[index - 1] + 1)
    if units[-1] > 999900:
        units[-1] = 999900
        for index in range(len(units) - 2, -1, -1):
            units[index] = min(units[index], units[index + 1] - 1)
    return [int(unit / 10000) if unit % 10000 == 0 else unit / 10000 for unit in units]


def forecast_case(case, name="case"):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} has unsupported quantile levels")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")

    reads = case["history"][-HISTORY_WINDOW:]
    read_times = [parse_time(read["at"]) for read in reads]
    origin = read_times[0]
    elapsed_minutes = [(instant - origin).total_seconds() / 60.0 for instant in read_times]
    all_xs = [math.log1p(max(0.0, minutes)) for minutes in elapsed_minutes]
    target_minutes = max(0.0, (parse_time(case["target"]) - origin).total_seconds() / 60.0)
    target_x = math.log1p(target_minutes)
    observed_span = max(0.0, (read_times[-1] - read_times[0]).total_seconds() / 60.0)

    fits = []
    last_rates = []
    for trophy_index in range(len(case["trophies"])):
        samples = [
            (x, logit(read["rates"][trophy_index]))
            for x, read in zip(all_xs, reads)
            if read["rates"][trophy_index] is not None
        ]
        if not samples:
            samples = [(all_xs[-1], logit(0.01))]
        xs = [sample[0] for sample in samples]
        ys = [sample[1] for sample in samples]
        fit = linear_fit(xs, ys)
        stationary = max(ys) - min(ys) < 1e-6
        fits.append((*fit, len(ys), observed_span, stationary, ys[-1]))
        last_rates.append(inv_logit(ys[-1]))

    ranked_rates = sorted(last_rates)
    lower_cut = ranked_rates[len(ranked_rates) // 3]
    upper_cut = ranked_rates[2 * len(ranked_rates) // 3]
    pool_members = {}
    pool_keys = []
    for trophy, rate, fit in zip(case["trophies"], last_rates, fits):
        band = 0 if rate <= lower_cut else 1 if rate <= upper_cut else 2
        key = (trophy["tier"], band)
        pool_keys.append(key)
        pool_members.setdefault(key, []).append(fit[1])
    pooled_slopes = {key: sum(values) / len(values) for key, values in pool_members.items()}

    horizon = max(0.0, float(case["horizon_minutes"]))
    rows = []
    for trophy, fit, key in zip(case["trophies"], fits, pool_keys):
        intercept, slope, residual_sd, r_squared, count, span, stationary, last_value = fit
        if stationary:
            prediction = last_value
            sigma_eff = 0.05
        else:
            calibrated_slope = evidence_weighted_slope(slope, pooled_slopes[key], count, r_squared)
            prediction = intercept + calibrated_slope * target_x
            prediction = min(last_value + 2.0, max(last_value - 2.0, prediction))
            safe_span = max(span, 1e-9)
            sigma_eff = math.sqrt(
                max(residual_sd, 0.08) ** 2 * (1.0 + horizon / safe_span)
                + (0.05 * math.sqrt(horizon / 60.0)) ** 2
            )
        projected = [clamp_rate(inv_logit(prediction + z * sigma_eff)) for z in Z_SCORES]
        rows.append({"id": trophy["id"], "q": valid_quantiles(projected)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(input_path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, input_path.name)
        output_path = OUTPUT_DIR / f"{input_path.stem}.out"
        output_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
