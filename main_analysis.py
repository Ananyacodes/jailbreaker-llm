import json
import os

from data.generate_synthetic_campaign import generate
from src.analysis import comparison, compliance_hmm, eda, severity_regression, state_classification, strategy_clustering, unsupervised


def main():
    print("STEP 1/8: generating synthetic campaign transcript dataset")
    rows = generate()
    os.makedirs("data/transcripts", exist_ok=True)
    out_path = "data/transcripts/synthetic_campaign_pbl.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  -> {len(rows)} turn-rows written to {out_path}")

    print("\nSTEP 2/8: exploratory data analysis")
    eda.main()

    print("\nSTEP 3/8: regression (predict eval-judge score from live-judge signal)")
    severity_regression.main()

    print("\nSTEP 4/8: classification (predict compliance state from turn metadata)")
    state_classification.main()

    print("\nSTEP 5/8: clustering (discover compliance state from judge scores alone)")
    strategy_clustering.main()

    print("\nSTEP 6/8: Review 2 PCA, K-means, GMM-EM and hierarchical clustering")
    unsupervised.main()

    print("\nSTEP 7/8: HMM (decode compliance state using judge scores + turn order)")
    compliance_hmm.main()

    print("\nSTEP 8/8: comparative performance analysis")
    comparison.main()

    print("\nAll done. See outputs/analysis/ for every plot and outputs/analysis/metrics.json for every number.")


if __name__ == "__main__":
    main()
