import json as json_codec
import math as numbers
import os as environment
from datetime import datetime as DateTime
from pathlib import Path as FilePath


QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_OFFSETS = [-1.645, -0.674, 0.0, 0.674, 1.645]
MAX_READS = 12
POOLING_STRENGTH = 0.30


def bounded_rate(number):
    return max(0.01, min(99.99, float(number)))


def rate_to_log_odds(rate):
    percent = bounded_rate(rate)
    return numbers.log(percent / (100.0 - percent))


def log_odds_to_rate(log_odds):
    if log_odds < 0.0:
        odds = numbers.exp(log_odds)
        return 100.0 * odds / (1.0 + odds)
    inverse_odds = numbers.exp(-log_odds)
    return 100.0 / (1.0 + inverse_odds)


def timestamp(text):
    return DateTime.fromisoformat(text.replace("Z", "+00:00"))


def least_squares(points):
    count = len(points)
    if count <= 1:
        return {"constant": points[-1][1], "gradient": 0.0, "sd": 0.0, "quality": 0.0}
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    spread_x = sum((x - mean_x) ** 2 for x, _ in points)
    if spread_x < 1e-15:
        return {"constant": mean_y, "gradient": 0.0, "sd": 0.0, "quality": 0.0}
    gradient = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread_x
    constant = mean_y - gradient * mean_x
    squared_error = sum((y - constant - gradient * x) ** 2 for x, y in points)
    sd = numbers.sqrt(squared_error / (count - 2)) if count > 2 else 0.0
    total_variation = sum((y - mean_y) ** 2 for _, y in points)
    quality = 1.0 - squared_error / total_variation if total_variation > 0.0 else 0.0
    return {
        "constant": constant,
        "gradient": gradient,
        "sd": sd,
        "quality": max(0.0, min(1.0, quality)),
    }


def assign_pool(tier, rate, first_boundary, second_boundary):
    if rate <= first_boundary:
        band = 0
    elif rate <= second_boundary:
        band = 1
    elif rate > second_boundary:
        band = 2
    return tier, band


def pooled_gradient_map(trophies, recent_rates, models):
    """Average slopes within each trophy-tier and current-rate tercile pool."""
    ordered = sorted(recent_rates)
    lower = ordered[len(ordered) // 3]
    upper = ordered[2 * len(ordered) // 3]
    keys = [
        assign_pool(trophy["tier"], rate, lower, upper)
        for trophy, rate in zip(trophies, recent_rates)
    ]
    buckets = {}
    for key, model in zip(keys, models):
        buckets.setdefault(key, []).append(model["gradient"])
    means = {key: sum(slopes) / len(slopes) for key, slopes in buckets.items()}
    return keys, means


def combine_gradient(own_gradient, group_gradient, observations, quality):
    evidence = max(0, observations - 2) * max(0.0, quality)
    group_share = POOLING_STRENGTH / (1.0 + evidence)
    return own_gradient * (1.0 - group_share) + group_gradient * group_share


def distinct_four_place_rates(raw_rates):
    ticks = [max(100, min(999900, numbers.floor(rate * 10000.0 + 0.5))) for rate in raw_rates]
    for position in range(1, len(ticks)):
        ticks[position] = max(ticks[position], ticks[position - 1] + 1)
    if ticks[-1] > 999900:
        ticks[-1] = 999900
        for position in range(len(ticks) - 2, -1, -1):
            ticks[position] = min(ticks[position], ticks[position + 1] - 1)
    return [tick // 10000 if tick % 10000 == 0 else tick / 10000 for tick in ticks]


def make_forecast(case, label):
    if case.get("levels") != QUANTILE_LEVELS:
        raise ValueError(f"{label} has unsupported quantile levels")
    if len(case.get("history", [])) == 0:
        raise ValueError(f"{label} has no history")

    reads = case["history"][-MAX_READS:]
    moments = [timestamp(read["at"]) for read in reads]
    start = moments[0]
    minute_marks = [(moment - start).total_seconds() / 60.0 for moment in moments]
    transformed_time = [numbers.log1p(max(0.0, minute)) for minute in minute_marks]
    target_minutes = max(0.0, (timestamp(case["target"]) - start).total_seconds() / 60.0)
    target_time = numbers.log1p(target_minutes)
    models = []
    recent_rates = []
    for trophy_number, _ in enumerate(case["trophies"]):
        points = [
            (x, rate_to_log_odds(read["rates"][trophy_number]))
            for x, read in zip(transformed_time, reads)
            if read["rates"][trophy_number] is not None
        ]
        if not points:
            points = [(transformed_time[-1], rate_to_log_odds(0.01))]
        model = least_squares(points)
        values = [y for _, y in points]
        model["unchanged"] = max(values) - min(values) < 1e-6
        model["latest"] = values[-1]
        model["count"] = len(values)
        models.append(model)
        recent_rates.append(log_odds_to_rate(values[-1]))

    membership, group_means = pooled_gradient_map(case["trophies"], recent_rates, models)

    horizon = float(max(0.0, case["horizon_minutes"]))
    answers = []
    for trophy, model, group in zip(case["trophies"], models, membership):
        if model["unchanged"]:
            center = model["latest"]
            uncertainty = 0.6
        elif not model["unchanged"]:
            gradient = combine_gradient(
                model["gradient"], group_means[group], model["count"], model["quality"]
            )
            fitted_center = model["constant"] + gradient * target_time
            center = model["latest"] + 0.5 * (fitted_center - model["latest"])
            center = max(model["latest"] - 1.5, min(model["latest"] + 1.5, center))
            uncertainty = 0.9 + 0.55 * numbers.log10(1.0 + horizon / 60.0)
        raw_quantiles = [
            bounded_rate(log_odds_to_rate(center + offset * uncertainty))
            for offset in NORMAL_OFFSETS
        ]
        answers.append({"id": trophy["id"], "q": distinct_four_place_rates(raw_quantiles)})
    return {"trophies": answers}


def run():
    source = FilePath(environment.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    destination = FilePath(environment.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    destination.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(source.glob("*.in")):
        case = json_codec.loads(case_path.read_text(encoding="utf-8"))
        result = make_forecast(case, case_path.name)
        result_path = destination / f"{case_path.stem}.out"
        result_path.write_text(json_codec.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if "__main__" == __name__:
    run()
