"""
Subset-leakage curve: does a proper subset of the minimal_supporting_set
already leak S, even though the paper's own validation never checks this?

The paper's construction-time checks are:
  C2: each SINGLE essential tool, alone, must NOT support S.
  C3: the FULL minimal_supporting_set (all CTD essential tools together)
      DOES support S.

Nothing in between is checked. For a CTD=5 instance, C2 only rules out
each of the 5 tools individually leaking S; it says nothing about whether,
say, 2 or 3 of those 5 tools together already leak S well short of the
full "minimal" set. If they do, minimal_supporting_set isn't actually
minimal in a meaningful sense -- a smaller subset would already suffice.

This script tests that directly. For each CTD level n in {2,3,4,5} and
each subset size k from 1 to n (i.e. every proper subset size PLUS the full
minimal_supporting_set itself, as the k=n reference point where C3 says
leakage is expected), it draws ONE random k-subset of the essential tools
per instance (independently per (instance, k) -- not a nested chain),
reveals only that subset's return_data (no noise tools), runs Stage 2, and
scores the result with the paper's exact leakage judge AND an adapted
task-completion judge (see common.py).

The completion judge matters because for k < n, the model is expected to
lack enough information to actually complete the benign task -- so a low
completion rate at k < n is the "expected" outcome. The interesting failure
mode is if leakage happens anyway even when the task ISN'T completed: that
would mean the model discloses S as a byproduct of reasoning about
insufficient evidence, not as a natural consequence of successfully doing
the task.

Results are grouped by (ctd, k) so you can compare, e.g., whether CTD=5
instances leak more at k=2 than CTD=3 instances do at k=2, and whether
leakage at k=n lines up with completion also being high there.
"""

import argparse
import json
import random
from pathlib import Path

from common import (
    DATA_PATH,
    RESULTS_DIR,
    Client,
    add_client_args,
    load_instances_by_ctd,
    run_completion_judge,
    run_judge,
    run_stage2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-ctd", type=int, default=10,
                     help="number of instances to sample per CTD level")
    ap.add_argument("--ctd-levels", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS_DIR / "subset_curve_raw.json"))
    add_client_args(ap)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sample = []
    for ctd in args.ctd_levels:
        pool = load_instances_by_ctd(DATA_PATH, ctd=ctd)
        rng.shuffle(pool)
        chosen = pool[: args.n_per_ctd]
        print(f"CTD={ctd}: {len(pool)} available, using {len(chosen)}")
        sample.extend(chosen)

    agent_client = Client(args.base_url, args.api_key, args.agent_model)
    judge_client = Client(args.base_url, args.api_key, args.judge_model)

    results = []
    for idx, instance in enumerate(sample):
        tools_by_id = {t["tool_id"]: t for t in instance["available_tools"]}
        essential_ids = instance["minimal_supporting_set"]
        ctd = len(essential_ids)

        print(f"[{idx + 1}/{len(sample)}] {instance['scenario_input_id']} ctd={ctd}")
        instance_result = {
            "scenario_input_id": instance["scenario_input_id"],
            "sensitive_conclusion": instance["sensitive_conclusion"],
            "ctd": ctd,
            "subsets": {},
        }
        for k in range(1, ctd + 1):  # proper subsets 1..ctd-1, plus the full set at k=ctd
            subset_rng = random.Random(f"{instance['scenario_input_id']}_k{k}")
            subset_ids = subset_rng.sample(essential_ids, k)
            tools = [tools_by_id[tid] for tid in subset_ids]
            try:
                answer = run_stage2(agent_client, instance["benign_user_goal"], tools)
                leak_verdict = run_judge(judge_client, instance["sensitive_conclusion"], answer)
                comp_verdict = run_completion_judge(judge_client, instance["benign_user_goal"], answer)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR on k={k}: {e}")
                answer = ""
                leak_verdict = {"answer": "error", "reason": str(e)}
                comp_verdict = {"answer": "error", "reason": str(e)}
            leaked = leak_verdict.get("answer") == "yes"
            completed = comp_verdict.get("answer") == "yes"
            print(f"  k={k}/{ctd} subset={subset_ids}: leaked={leaked} completed={completed} "
                  f"({leak_verdict.get('reason', '')[:60]})")
            instance_result["subsets"][f"k{k}"] = {
                "k": k,
                "tool_ids": subset_ids,
                "model_response": answer,
                "judge_verdict": leak_verdict,
                "leaked": leaked,
                "completion_verdict": comp_verdict,
                "completed": completed,
            }
        results.append(instance_result)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote raw results to {args.out}")


if __name__ == "__main__":
    main()
