`build_demo_assets.py` creates deterministic placeholder-based datasets for
testing the judge training and end-to-end pipeline. These rows deliberately do
not contain real harmful procedures and are not a production evaluation set.

`seed_prompts.txt` contains high-level authorized local safety probes. Replace
or extend it with prompts that match your own model policy and evaluation
scope before interpreting campaign metrics.
