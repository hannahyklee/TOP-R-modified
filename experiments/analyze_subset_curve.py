"""Aggregate subset_curve_raw.json into leakage rate AND completion rate vs.
subset size k, one line per CTD level, to check whether proper subsets of
the minimal_supporting_set already leak S (a gap the paper's own C2/C3
validation never checks), and whether leakage happens even when the task
isn't actually completed."""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt

from common import RESULTS_DIR

COLORS = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868", 5: "#C44E52"}


def aggregate(results, field):
    # cells[ctd][k] = list of bools
    cells = defaultdict(lambda: defaultdict(list))
    for inst in results:
        ctd = inst["ctd"]
        for step_key, step in inst["subsets"].items():
            cells[ctd][step["k"]].append(step[field])
    rates = {ctd: {k: sum(v) / len(v) for k, v in ks.items()} for ctd, ks in cells.items()}
    counts = {ctd: {k: len(v) for k, v in ks.items()} for ctd, ks in cells.items()}
    return rates, counts


def plot_panel(ax, rates, counts, title):
    for ctd in sorted(rates.keys()):
        ks = sorted(rates[ctd].keys())
        values = [rates[ctd][k] * 100 for k in ks]
        color = COLORS.get(ctd, "gray")
        ax.plot(ks, values, linewidth=2, color=color, label=f"CTD={ctd}")
        # proper subsets as circles, the full-set (C3) point as a star
        ax.plot(ks[:-1], values[:-1], marker="o", linewidth=0, color=color)
        ax.plot(ks[-1], values[-1], marker="*", markersize=14, linewidth=0, color=color)
        for k, v in zip(ks, values):
            ax.annotate(f"{v:.0f}%", (k, v),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    max_k = max(k for ks in rates.values() for k in ks)
    ax.set_xticks(range(1, max_k + 1))
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def plot(leak_rates, leak_counts, comp_rates, comp_counts, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9), sharex=True)
    plot_panel(ax1, leak_rates, leak_counts, "Leakage rate vs. subset size k (★ = full set / C3 point)")
    plot_panel(ax2, comp_rates, comp_counts, "Task-completion rate vs. subset size k")
    ax2.set_xlabel("Subset size k (k = CTD is the full minimal_supporting_set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(RESULTS_DIR / "subset_curve_raw.json"))
    ap.add_argument("--plot-out", default=str(RESULTS_DIR / "subset_curve.png"))
    args = ap.parse_args()

    with open(args.infile) as f:
        results = json.load(f)

    leak_rates, leak_counts = aggregate(results, "leaked")
    comp_rates, comp_counts = aggregate(results, "completed")
    plot(leak_rates, leak_counts, comp_rates, comp_counts, args.plot_out)

    print()
    for ctd in sorted(leak_rates.keys()):
        print(f"CTD={ctd}:")
        for k in sorted(leak_rates[ctd].keys()):
            marker = " <- full set (C3 point)" if k == ctd else ""
            print(f"  k={k}/{ctd}: leaked={leak_rates[ctd][k]*100:.1f}% "
                  f"completed={comp_rates[ctd][k]*100:.1f}% (n={leak_counts[ctd][k]}){marker}")


if __name__ == "__main__":
    main()
