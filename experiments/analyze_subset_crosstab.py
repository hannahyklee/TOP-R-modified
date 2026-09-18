"""Cross-tabulate leaked x completed for every (ctd, k) cell in
subset_curve_raw.json, to check whether leakage co-occurs with task
completion or happens independently of (or despite) it."""

import argparse
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout

from common import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(RESULTS_DIR / "subset_curve_raw.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "subset_crosstab.txt"))
    args = ap.parse_args()

    with open(args.infile) as f:
        results = json.load(f)

    # cells[(ctd, k)] = {"LC": n, "LnC": n, "nLC": n, "nLnC": n}
    cells = defaultdict(lambda: {"leak_and_complete": 0, "leak_only": 0,
                                  "complete_only": 0, "neither": 0})
    for inst in results:
        ctd = inst["ctd"]
        for step in inst["subsets"].values():
            k = step["k"]
            leaked, completed = step["leaked"], step["completed"]
            key = (ctd, k)
            if leaked and completed:
                cells[key]["leak_and_complete"] += 1
            elif leaked and not completed:
                cells[key]["leak_only"] += 1
            elif not leaked and completed:
                cells[key]["complete_only"] += 1
            else:
                cells[key]["neither"] += 1

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Cross-tab of leaked x completed per (CTD, k) cell.")
        print("leak_only = leaked but did NOT complete the task; "
              "leak_and_complete = leaked while also completing it.\n")
        print(f"{'CTD':>4} {'k':>3} {'n':>4}  {'leak&complete':>14} {'leak_only':>10} "
              f"{'complete_only':>14} {'neither':>8}   note")
        total_leak = 0
        total_leak_only = 0
        for (ctd, k), c in sorted(cells.items()):
            n = sum(c.values())
            note = " <- full set" if k == ctd else ""
            print(f"{ctd:>4} {k:>3} {n:>4}  {c['leak_and_complete']:>14} {c['leak_only']:>10} "
                  f"{c['complete_only']:>14} {c['neither']:>8}{note}")
            total_leak += c["leak_and_complete"] + c["leak_only"]
            total_leak_only += c["leak_only"]

        if total_leak:
            pct = 100 * total_leak_only / total_leak
            print(f"\nAcross all cells: {total_leak_only}/{total_leak} "
                  f"({pct:.0f}%) of leaks happened WITHOUT task completion.")

        # Same breakdown restricted to k < ctd (proper subsets only, excluding the C3 full-set point)
        proper_leak = 0
        proper_leak_only = 0
        for (ctd, k), c in cells.items():
            if k < ctd:
                proper_leak += c["leak_and_complete"] + c["leak_only"]
                proper_leak_only += c["leak_only"]
        if proper_leak:
            pct = 100 * proper_leak_only / proper_leak
            print(f"Proper subsets only (k<CTD): {proper_leak_only}/{proper_leak} "
                  f"({pct:.0f}%) of leaks happened WITHOUT task completion.")

    text = buf.getvalue()
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"Saved cross-tab to {args.out}")


if __name__ == "__main__":
    main()
