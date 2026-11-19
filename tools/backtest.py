#!/usr/bin/env python3
"""One-step backtest over reads three and four of both practice cases."""

import copy
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import LEVELS, Z_SCORES, forecast_case, logit, parse_time  # noqa: E402


def pinball(actual, predicted, level):
    error = actual - predicted
    return level * error if error >= 0.0 else (level - 1.0) * error


def persistence_predictions(case, sigma):
    return [
        [logit(rate) + z_score * sigma for z_score in Z_SCORES]
        for rate in case["history"][-1]["rates"]
    ]


def adaptive_predictions(case):
    output = forecast_case(case, "backtest")
    return [[logit(value) for value in row["q"]] for row in output["trophies"]]


def score(predictions, actual_rates):
    losses = []
    for predicted, rate in zip(predictions, actual_rates):
        actual = logit(rate)
        losses.extend(pinball(actual, quantile, level) for quantile, level in zip(predicted, LEVELS))
    return sum(losses) / len(losses)


def truncated_case(source, read_count):
    case = copy.deepcopy(source)
    next_read = source["history"][read_count]
    previous_read = source["history"][read_count - 1]
    case["history"] = source["history"][:read_count]
    case["target"] = next_read["at"]
    case["horizon_minutes"] = (
        parse_time(next_read["at"]) - parse_time(previous_read["at"])
    ).total_seconds() / 60.0
    return case, next_read["rates"]


def main():
    paths = (ROOT / "p00.in", ROOT / "p01.in")
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results = {name: {3: [], 4: []} for name in ("starter persistence", "current fixed", "adaptive")}

    for source in cases:
        for read_count in (3, 4):
            case, actual = truncated_case(source, read_count)
            horizon = case["horizon_minutes"]
            results["starter persistence"][read_count].append(
                score(persistence_predictions(case, 0.3), actual)
            )
            fixed_sigma = math.sqrt(0.15 ** 2 + (0.06 * math.sqrt(horizon / 60.0)) ** 2)
            results["current fixed"][read_count].append(
                score(persistence_predictions(case, fixed_sigma), actual)
            )
            results["adaptive"][read_count].append(score(adaptive_predictions(case), actual))

    print(f"{'model':24} {'cut 3':>10} {'cut 4':>10} {'mean':>10}")
    for name, cuts in results.items():
        cut_three = sum(cuts[3]) / len(cuts[3])
        cut_four = sum(cuts[4]) / len(cuts[4])
        print(f"{name:24} {cut_three:10.6f} {cut_four:10.6f} {(cut_three + cut_four) / 2:10.6f}")


if __name__ == "__main__":
    main()
