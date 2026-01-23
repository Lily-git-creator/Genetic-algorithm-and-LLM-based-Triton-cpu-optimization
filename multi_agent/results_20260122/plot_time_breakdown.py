#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Line chart: absolute time (seconds) of API call & Evaluator vs Generation.

You only need to fill:
  - generations
  - api_time_s
  - evaluator_time_s

Then run:
  python plot_time_vs_gen_manual.py
"""

from typing import List
import matplotlib.pyplot as plt


def validate(generations: List[int], api_time_s: List[float], evaluator_time_s: List[float]) -> None:
    if not (len(generations) == len(api_time_s) == len(evaluator_time_s)):
        raise ValueError(
            f"Length mismatch: generations={len(generations)}, "
            f"api_time_s={len(api_time_s)}, evaluator_time_s={len(evaluator_time_s)}"
        )
    if len(generations) == 0:
        raise ValueError("Empty data: please fill generations/api_time_s/evaluator_time_s.")
    if any(g2 <= g1 for g1, g2 in zip(generations, generations[1:])):
        raise ValueError("generations must be strictly increasing (e.g., [0,1,2,...]).")


def main():
    # =========================
    # TODO: Fill your data here
    # =========================
    generations = [
        1, 2, 3, 4, 5

    ]

    # Absolute time in seconds for API call per generation
    api_time_s = [
        334.53, 472.07,618.6, 783.99, 998.92
    ]

    # Absolute time in seconds for Evaluator per generation
    evaluator_time_s = [
        70.35, 425.99, 481.74, 726.94, 835.44
    ]
    # =========================
    # End of manual input
    # =========================

    validate(generations, api_time_s, evaluator_time_s)

    # Plot
    plt.figure()
    plt.plot(generations, api_time_s, marker="o", linestyle="-", label="API call (Total s)")
    plt.plot(generations, evaluator_time_s, marker="s", linestyle="--", label="Evaluator (Total s)")

    plt.xlabel("Generation")
    plt.ylabel("Absolute time (s)")
    plt.title("Absolute Time vs Generation")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    # Save (and also show if you want)
    out_path = "time_vs_generation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved to: {out_path}")
    # plt.show()


if __name__ == "__main__":
    main()
