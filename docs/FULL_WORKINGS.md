# Jailbreaker LLM: full workings and verified local workflow

This document describes the complete local pipeline, the changes made during
the performance/reliability pass, and the exact smoke test that was verified on
an NVIDIA GeForce RTX 3050 6 GB Laptop GPU.

## 1. What the system does

The project is an authorized LLM safety-evaluation pipeline with four distinct
model roles:

1. **Attacker policy** — Qwen2.5-0.5B-Instruct plus trainable LoRA adapters and
   a value head. It generates candidate evaluation prompts.
2. **Target model** — the local model being evaluated. The included smoke setup
   uses the already-cached SmolLM2-135M checkpoint under `checkpoints/base_model`.
3. **Reward judge** — a DistilBERT classifier used only to supply PPO rewards.
4. **Evaluation judge** — a separately trained DistilBERT classifier used only
   after a campaign. It does not provide training gradients or rewards.

The included data and checkpoints prove that every component connects and
runs. The generated judge data uses explicit placeholders such as
`[PROCEDURAL_STEP_1]`; it contains no real harmful procedure. These judges are
integration fixtures, not production safety evaluators.

## 2. End-to-end data flow

```text
seed prompts
    |
    v
Qwen attacker + LoRA -----> generated evaluation prompts
    |                              |
    |                              v
    |                       authorized local target
    |                              |
    |                              v
    +<----- reward judge scores target response
    |
    v
PPO: reward + KL penalty -> GAE -> clipped policy/value update
    |
    v
compact LoRA adapter + value-head checkpoint
    |
    v
multi-turn campaign -> independent eval judge -> redaction -> JSONL + report
```

## 3. Asset preparation

Run:

```powershell
python data/build_demo_assets.py
```

This deterministically creates:

- `data/judge_train/train.jsonl`: 144 reward-judge training rows.
- `data/judge_train/val.jsonl`: 36 reward-judge validation rows.
- `data/eval_judge_train/train.jsonl`: 144 separate eval-judge training rows.
- `data/eval_judge_train/val.jsonl`: 36 separate eval-judge validation rows.

The repository also includes:

- `data/seed_prompts.txt`: six high-level authorized safety probes.
- `data/seed_prompts_smoke.txt`: the single-seed fast integration run.

The two judge datasets use different response phrase banks and random seeds so
the reward and evaluation checkpoints are not the same model artifact.

## 4. Judge training

Reward judge:

```powershell
python -m src.judge.train_judge `
  --train data/judge_train/train.jsonl `
  --val data/judge_train/val.jsonl `
  --out_dir checkpoints/judge_model `
  --epochs 1 --batch_size 16 --max_len 128 --seed 42
```

Independent evaluation judge:

```powershell
python -m src.judge.train_judge `
  --train data/eval_judge_train/train.jsonl `
  --val data/eval_judge_train/val.jsonl `
  --out_dir checkpoints/eval_judge_model `
  --epochs 1 --batch_size 16 --max_len 128 --seed 84
```

Each checkpoint contains a DistilBERT encoder and a three-class classifier:

- `refusal`
- `partial_compliance`
- `full_violation`

The smoke run reached 100% validation accuracy for both checkpoints. This is
expected on the intentionally obvious template data and must not be interpreted
as real-world judge accuracy.

## 5. Attacker model and LoRA

The attacker base is `Qwen/Qwen2.5-0.5B-Instruct`. The base remains in the
Hugging Face cache; the project does not create another full copy.

Only these components are trained:

- LoRA adapters on attention and MLP projections.
- The scalar value head used by PPO.

The reference policy shares the same frozen Qwen base. During reference
scoring, adapters are temporarily disabled. This avoids loading a second 0.5B
reference model into VRAM.

The verified model contained approximately 502.8 million total parameters and
8.8 million trainable parameters, or 1.75%.

## 6. PPO training internals

For every rollout batch:

1. Seed prompts are encoded using Qwen's chat template.
2. Attacker generations are produced in one left-padded batch.
3. The target receives the generated prompts in configurable batches.
4. The reward judge assigns refusal, partial-compliance, and violation scores.
5. Scores are converted to a scalar reward. Degenerate text is penalized.
6. The policy and reference log-probabilities are evaluated in micro-batches.
7. A per-token KL penalty discourages excessive drift from the base policy.
8. Generalized Advantage Estimation computes advantages and returns.
9. PPO applies its clipped policy objective plus value loss.
10. LoRA and value-head weights are updated and checkpointed.

Numerically sensitive probability, KL, advantage, and value-loss calculations
run in FP32 even though the transformer runs in BF16 on the tested GPU.

`rl.ppo_micro_batch_size` controls the speed/memory trade-off. Four was
verified on the 6 GB RTX 3050. Reduce it to two or one if substantially longer
prompts exhaust memory.

Run the verified two-rollout PPO job with:

```powershell
python -m src.attacker.rl_finetune --config configs/smoke_test.yaml
```

It writes:

- `checkpoints/attacker_smoke/adapter_model.safetensors`
- `checkpoints/attacker_smoke/adapter_config.json`
- `checkpoints/attacker_smoke/value_head.pt`
- `checkpoints/attacker_smoke/trainer_state.pt`

The adapter itself is about 33.6 MB. The larger trainer-state file holds AdamW
optimizer state for resuming training.

## 7. Target execution

The smoke configuration uses:

```yaml
target:
  mode: local
  local_model_name: checkpoints/base_model
  local_tokenizer_name: checkpoints/base_model/tokenizer
```

Model and tokenizer locations may differ. The target now accepts both paths.
Generation uses hardware-aware BF16/FP16, PyTorch SDPA attention, KV caching,
left truncation, and batched candidate execution.

For each multi-turn step, the agent tries multiple techniques, sends all valid
branches to the target in a batch, judges the responses, and keeps the best
branch. The winning target response is reused as the next committed response;
it is not generated again. No unused branch batch is produced after the final
turn.

## 8. Campaign and reporting

Run:

```powershell
python -m src.eval.run_campaign `
  --config configs/smoke_test.yaml `
  --seeds data/seed_prompts_smoke.txt `
  --campaign_id smoke_e2e
```

The campaign:

1. Reloads the saved Qwen LoRA adapter.
2. Loads the reward judge for live multi-turn branch selection.
3. Loads the separate evaluation judge for final scoring.
4. Loads the authorized local target.
5. Runs the configured turn/branch limits.
6. Redacts recognizable sensitive-data patterns.
7. Writes `data/transcripts/smoke_e2e.jsonl`.
8. Writes `data/transcripts/smoke_e2e_report.json`.

The verified smoke campaign ran one seed for two turns. Its report contained:

- 2 recorded attempts.
- 0% full-violation rate.
- 0% partial-compliance rate.
- 100% refusal final-label rate.
- 0 conversations flagged for judge disagreement.
- `single_judge_fallback: true`, because no external LLM-as-judge endpoint was
  configured. The evaluation classifier is still separate from the reward
  classifier.

## 9. One-command Windows validation

From the repository root:

```powershell
.\scripts\run_smoke_test.ps1
```

If both judge checkpoints already exist:

```powershell
.\scripts\run_smoke_test.ps1 -SkipJudgeTraining
```

The script builds demo data, trains both judges unless skipped, runs PPO, then
runs the campaign and report generator.

## 10. Performance changes made

### Model quality and memory

- Replaced the earlier tiny attacker with Qwen2.5-0.5B-Instruct.
- Added LoRA instead of full-model PPO updates.
- Shared the frozen base with the adapter-disabled reference policy.
- Saved compact adapters instead of duplicating the base model.
- Reused the Hugging Face base-model cache to avoid another roughly 1 GB copy.

### Generation speed

- Batched Qwen attacker rollouts.
- Batched local-target best-of-N branches.
- Added SDPA, KV caching, context caps, and hardware-aware precision.
- Reused the selected branch response instead of generating it twice.
- Removed the unused final-turn branch generation.

On the tested GPU, eight 16-token attacker generations improved from 21.52
seconds serially to 4.05 seconds batched, a 5.31x speedup.

### PPO speed and stability

- Batched policy and reference scoring with configurable micro-batches.
- Kept PPO statistics and losses in FP32 under BF16 model execution.
- Optimized only parameters with `requires_grad=True`.
- Disabled dropout during rollout/old-policy scoring and restored training mode
  during optimization.
- Added batch/reward length validation and stable single-token standard
  deviation handling.

For eight 16-token samples, PPO improved from 2.786 seconds with one-sample
passes to 0.720 seconds with micro-batches of four, a 3.87x speedup. Peak
allocated VRAM increased from 1.28 GB to 1.90 GB.

### Reliability and integration

- Added Hugging Face chat-template encoding and attention masks.
- Added adapter save/reload and resumable optimizer state.
- Made experiment tracking opt-in so local runs never wait for a login.
- Avoided loading the sentence-transformer diversity model when diversity is
  disabled.
- Added separate model/tokenizer paths for local targets.
- Separated reward-judge and eval-judge checkpoint paths.
- Fixed the RL entry point passing the wrong configuration level to the reward
  function; this bug was found by the first complete smoke run.
- Added deterministic judge training seeds and configurable batch size,
  sequence length, and learning rate.
- Added Windows smoke automation.

## 11. Verification record

Verified locally on 2026-09-25:

- 45 automated tests passed.
- Reward judge trained and saved successfully.
- Independent evaluation judge trained and saved successfully.
- Qwen attacker generated two rollouts.
- Local Smol target generated batched responses.
- Reward scores were converted and consumed by PPO.
- BF16 forward/backward and optimizer step completed.
- LoRA adapter, value head, and trainer state saved.
- Saved adapter reloaded by the campaign runner.
- Two-turn multi-branch campaign completed.
- Redacted JSONL transcript and JSON report were written.

## 12. What must change for a meaningful evaluation

The included smoke artifacts validate software plumbing only. Before using
campaign metrics for decisions:

1. Replace the placeholder judge datasets with reviewed, policy-specific,
   independently labeled examples.
2. Train reward and evaluation judges on genuinely disjoint datasets and
   measure per-class precision, recall, calibration, and confusion matrices.
3. Configure an independent LLM-as-judge or human review workflow for final
   evaluation; do not use it as the PPO reward judge.
4. Replace the 135M smoke target with the actual local model or explicitly
   authorized API target under evaluation.
5. Replace high-level smoke seeds with a scoped evaluation suite and retain
   authorization records.
6. Increase rollout count gradually (for example 100, then 1,000) while
   monitoring reward curves, KL, diversity, latency, and held-out success.
7. Review transcript redaction before exporting data; pattern-based redaction
   is not a guarantee against every form of sensitive information.

Do not interpret the synthetic judges' 100% validation accuracy or the smoke
campaign's refusal rate as evidence about a production model's safety.
