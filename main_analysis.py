import json
import os

from data.generate_synthetic_campaign import generate
from src.analysis import comparison, compliance_hmm, eda, severity_regression, state_classification, strategy_clustering


def main():
    print("STEP 1/7: generating synthetic campaign transcript dataset")
    rows = generate()
    os.makedirs("data/transcripts", exist_ok=True)
    out_path = "data/transcripts/synthetic_campaign_pbl.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  -> {len(rows)} turn-rows written to {out_path}")

    print("\nSTEP 2/7: exploratory data analysis")
    eda.main()

    print("\nSTEP 3/7: regression (predict eval-judge score from live-judge signal)")
    severity_regression.main()

    print("\nSTEP 4/7: classification (predict compliance state from turn metadata)")
    state_classification.main()

    print("\nSTEP 5/7: clustering (discover compliance state from judge scores alone)")
    strategy_clustering.main()

    print("\nSTEP 6/7: HMM (decode compliance state using judge scores + turn order)")
    compliance_hmm.main()

    print("\nSTEP 7/7: comparative performance analysis")
    comparison.main()

    print("\nAll done. See outputs/analysis/ for every plot and outputs/analysis/metrics.json for every number.")


if __name__ == "__main__":
    main()
