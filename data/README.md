# data/

Nothing here is shipped populated. This is what goes where:

- `corpus/*.txt` -- only needed for a from-scratch pretrain (see
  configs/base_model.yaml: `pretrain.warm_start_from`). Any plain-text
  corpus you have rights to train on.
- `judge_train/raw.jsonl` -- pull a public red-teaming benchmark with
  binary refusal/compliance labels (e.g. JailbreakBench, AdvBench) and
  point `scripts/relabel_for_judge.py` at it to produce
  `judge_train/train.jsonl` / `judge_train/val.jsonl`. Train the Reward
  Judge and the Eval Judge's classifier half from two different splits (or
  at least two different seeds) -- see that script's docstring.
- `seed_prompts.txt` -- one opening prompt per line, `#` for comments. Not
  attacks themselves, just conversation starters the multi-turn agent
  escalates from.
- `transcripts/` -- campaign output, git-ignored. Created automatically by
  `src/eval/run_campaign.py`.

Do not point any of this at a target you don't own, control, or have
explicit written authorization to test. See the top-level README's
"responsible use" section.
