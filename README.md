# jailbreaker-llm

An LLM-vs-LLM automated red-teaming research platform. An attacker model,
warm-started from an existing open-weight base and then fine-tuned via
reinforcement learning, learns to discover prompts and multi-turn
conversation strategies that cause a *target* LLM to violate its own
policies -- with an independent judge ensemble scoring outcomes and a
mutation module mining which strategies transfer across targets.

This is LLM engineering first, security second: the bulk of the work is
RL fine-tuning (custom PPO), judge-model training, and multi-turn agent
design. The security framing is what the trained system is pointed at.

This repo intentionally stops at a generic scored vulnerability report --
there is no vendor-specific disclosure-ticket export here. If you need to
feed a specific vulnerability-management process, build that as a separate
step that reads `report.json`; don't fork this repo to add it.

## responsible use

This repo is for authorized safety evaluation only -- running it against a
model you own, control, or have explicit permission to test (your own
fine-tunes, local open-weight models, or a target you've been given
permission/API access to red-team). It is not configured with any working
exploits out of the box; `data/` includes only a placeholder-based integration
dataset builder and high-level local smoke seeds. Do not point the attacker loop at third-party production systems
without authorization -- that's the same rule as any other pentesting tool.

Enforcement, not just policy: `configs/allowlist.yaml` lists the only API
hosts the attacker/agent loop is permitted to call, and that check runs
INSIDE `APITarget.__init__` itself (`src/targets/allowlist.py`), not in one
factory function some call sites might skip. A circuit breaker halts a run
after repeated consecutive failures against a target instead of retrying
forever. See "architecture" below for the rest of the safety-relevant
design.

## architecture

```
 ┌─────────────────────────┐
 │  Base LM (warm-start)      │  src/model/pretrain.py --
 │  existing open-weight base,  │  configs/base_model.yaml:
 │  its own tokenizer             │  pretrain.warm_start_from
 └──────────────┬─────────────────┘
                ▼
 ┌─────────────────────────────┐        ┌────────────────────┐
 │  SimplePPOTrainer              │◄──────│ Reward Judge          │  src/judge/ --
 │  src/attacker/ppo_trainer.py   │       │ (train-only classifier)│  train-only, never
 │  generate -> judge -> GAE ->   │       └────────────────────┘  scores a final report
 │  clipped PPO update            │
 │  + diversity term              │  reward.py: judge score minus
 │  (src/attacker/reward.py)      │  embedding similarity to this
 └──────────────┬────────────────┘  run's recent successful attacks
                ▼
 ┌─────────────────────────────┐
 │   Multi-turn agent loop       │  src/agent/multi_turn_loop.py --
 │   technique menu (persona,      │  best-of-N branch search: try a
 │   hypothetical, obfuscation,      │  few techniques against the
 │   escalate/retreat) + tree-search   │  target, keep whichever the
 │   scores the FULL conversation        │  live judge scores highest
 │   so far, not just the latest turn      │
 └──────────────┬─────────────────────────┘
                ▼
     ┌─────────────────────┐
     │  Scope gate            │  src/targets/allowlist.py --
     │  allowlist + circuit     │  enforced inside
     │  breaker                   │  LocalTarget/APITarget.respond()
     └──────────┬─────────────────┘
                ▼
      ┌───────────────────┐
      │   Target LLM(s)    │  src/targets/ (local or API, retry+backoff+timeout)
      └──────────┬─────────┘
                 ▼
      ┌───────────────────┐
      │  Campaign runner   │  src/eval/run_campaign.py -- resumable,
      │                     │  writes an authorization ledger entry
      └──────────┬─────────┘  per row (target_mode/target_host)
                 ▼
     ┌─────────────────────┐
     │  Redacted transcript   │  src/eval/redact.py -- strips
     │  store                    │  PII/diagnostic-shaped data before
     └────────┬────────┬─────────┘  anything downstream reads a row
              ▼        ▼
 ┌────────────────┐   ┌─────────────────────────┐
 │  Eval Judge        │   │  Mutation/clustering       │
 │  classifier + LLM-    │   │  src/mutation/                │
 │  judge ensemble,        │   │  embedding clusters, not        │
 │  scores full transcript   │   │  TF-IDF; drops rows whose         │
 │  (src/judge/eval_judge.py)  │   │  target authorization expired       │
 └────────┬───────────────────┘   └─────────────────────────┘
          ▼
   ┌─────────────────────────┐
   │  Scored report              │  src/eval/report.py --
   │  (flags disagreement            │  generic, no ticket/disclosure
   │  between judges for human           │  export; build that
   │  review, doesn't auto-decide)         │  separately if you need it
   └─────────────────────────┘
```

Three design decisions worth calling out:

**The Reward Judge and Eval Judge are never the same model.** If the
attacker trained directly against the judge that later "proves" a finding
is real, PPO would happily learn to exploit that judge's blind spots
instead of finding real policy violations. The Reward Judge exists only to
produce a fast training signal; the Eval Judge that scores a completed
campaign is a differently-built ensemble (a separately-trained classifier
plus an LLM-as-judge) that never sees a gradient from the attacker it's
grading.

**The judge scores the conversation, not the last message.** A judge that
only sees the latest exchange can't recognize a violation that only makes
sense in light of what was set up two turns earlier -- which defeats the
purpose of a multi-turn agent. Both the live per-turn judge (windowed
context, `src/agent/multi_turn_loop.py::_score_with_context`) and the
eval-time ensemble (`src/judge/eval_judge.py`) score against the
conversation so far, not just the current turn.

**Warm-start, not pretrain-from-scratch, is the default.** A from-scratch
run is the biggest single time sink in this pipeline and isn't necessary
to get the actual differentiator (an attacker trained on reward from the
real target) -- see `configs/base_model.yaml`. Pretraining from scratch is
still supported (`pretrain.warm_start_from: null`) if you deliberately
want your own architecture and tokenizer; `src/tokenizer/` is only
relevant on that path.

The default warm start is now `Qwen/Qwen2.5-0.5B-Instruct`. PPO updates
LoRA adapters plus the value head instead of every base-model parameter.
The KL reference uses the same frozen base with adapters temporarily
disabled, so the 6 GB laptop GPU does not hold a second 0.5B model copy.
Saved attacker checkpoints contain the compact adapter and value head;
the unchanged base is reused directly from the Hugging Face cache. This avoids
an extra roughly 1 GB project copy on Windows. Set
`pretrain.copy_to_out_dir: true` only when a standalone base-model directory
is explicitly needed.

## local performance

Local inference is hardware-aware and configured under `target:`:

- `dtype: auto` uses BF16 on supported CUDA GPUs, FP16 on older CUDA GPUs,
  and FP32 on CPU.
- `attn_implementation: sdpa` and `use_cache=True` select PyTorch's
  optimized attention path and reuse the KV cache during generation.
- `max_input_tokens` bounds the growing multi-turn context; truncation keeps
  the newest turns.
- `batch_size` batches best-of-N target trials. The selected trial response
  is reused on the next turn instead of being generated a second time.
- Hugging Face attacker rollouts are also generated as one left-padded batch
  instead of looping over every PPO prompt serially.
- PPO policy/reference scoring and loss passes use configurable micro-batches
  (`rl.ppo_micro_batch_size`) to trade a small amount of VRAM for much faster
  training without requiring the full rollout batch to fit at once.
- No candidate batch is generated after the final allowed turn.

With the default six turns and three candidates, the old loop could perform
24 serial target generations (including duplicate winners and a final unused
branch). The new local path performs six generation rounds and evaluates 16
sequences. Actual wall-clock speedup depends on the target model and GPU.

Optional `quantization: 4bit` or `8bit` is available for CUDA targets after
installing `bitsandbytes`. Quantization is primarily a memory-saving option;
for a small model it may be slower than BF16.

## module map

| Path | What it does |
|---|---|
| `src/tokenizer/` | Trains a custom BPE tokenizer -- only used for a from-scratch run |
| `src/model/` | Base transformer (nanoGPT-style) + from-scratch pretraining loop; `pretrain.py` also handles warm-starting from an existing open-weight base |
| `src/attacker/value_head.py` | Adds a value head to either backend and supports a shared-base LoRA reference policy |
| `src/attacker/ppo_trainer.py` | Custom PPO (GAE + clipped surrogate + KL-to-reference penalty), advantages normalized across the batch |
| `src/attacker/reward.py` | Judge score -> scalar reward, plus an embedding-similarity diversity penalty so PPO doesn't collapse onto one trick |
| `src/attacker/technique_library.py` | Named attack techniques (persona/hypothetical framing, obfuscation, escalation) the multi-turn agent and mutation module both draw from |
| `src/attacker/rl_finetune.py` | The RL fine-tuning entry point -- resumable, redacts sensitive config before any wandb logging |
| `src/judge/train_judge.py`, `judge_infer.py` | Trains/runs the train-only Reward Judge classifier |
| `src/judge/llm_judge.py`, `eval_judge.py` | The independent eval-time judge: LLM-as-judge + a separately-trained classifier, combined into an ensemble that flags disagreement for human review |
| `src/agent/` | Multi-turn conversational loop: technique menu + best-of-N branch search, scores the full conversation so far |
| `src/targets/` | `TargetModel` interface, `LocalTarget`/`APITarget` (both scope-gated + circuit-breaker protected), shared `build_target` factory |
| `src/mutation/` | Generates variants of a successful attack via the technique library; clusters transcripts by embedding similarity (not TF-IDF) to find transferable patterns, dropping revoked targets |
| `src/eval/` | Runs a campaign (`run_campaign.py`, resumable), redacts transcripts (`redact.py`), produces a generic scored report (`report.py`) |
| `scripts/relabel_for_judge.py` | Converts a binary refusal/compliance benchmark into the judge's 3-class scheme |

## build order

1. `src/model/pretrain.py` -- warm-start by default; only touch `src/tokenizer/` if you deliberately choose a from-scratch run
2. `src/judge/train_judge.py` -- train the Reward Judge; train a SEPARATE checkpoint for the Eval Judge's classifier half from a different split/seed
3. `src/attacker/` -- RL fine-tuning, depends on (1) and (2)
4. `src/agent/` -- multi-turn behavior on top of the trained attacker
5. `src/eval/` -- run a campaign and get a scored report
6. `src/mutation/` -- once you have real transcripts, mine them for transferable strategies

See `scripts/` for the runnable order (`01_...sh` through `05_...sh`).

## setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, the complete local integration path is:

```powershell
.\scripts\run_smoke_test.ps1
```

See [`docs/FULL_WORKINGS.md`](docs/FULL_WORKINGS.md) for component-level
behavior, individual commands, verified measurements, every material change,
and the distinction between the synthetic smoke judges and a real evaluation.

## tests

```bash
pytest
```

Coverage focuses on the orchestration code that previously had none: the
multi-turn agent loop (including a regression test for a real crash bug --
see `tests/test_multi_turn_loop.py`'s docstring), the allowlist/circuit
breaker, and the technique library's branch-selection logic. Anything
requiring network access (downloading `distilbert-base-uncased`, a
sentence-transformers model, or a warm-start base) is left out of the
default test run by construction -- those paths are exercised by actually
running the pipeline, not by unit tests.

## known limitations

- **Single target per campaign.** Every config here points at exactly one
  target. A fleet/campaign-matrix concept (useful once you have more than
  one authorized target, e.g. multiple product versions) isn't built --
  add it once a single-target campaign is solid, not before.
- **The Eval Judge's classifier half inherits DistilBERT's 512-token
  window.** It's given the most recent turns that fit, not the oldest --
  but on a long conversation it is not seeing everything the LLM-judge half
  sees. A longer-context encoder or a summarize-then-score step would close
  this gap.
- **The judge training data is heuristically labeled.**
  `scripts/relabel_for_judge.py` synthesizes the `partial_compliance` class
  from regex/length heuristics. Its docstring says to spot-check ~50 rows
  before trusting it -- nothing enforces that happening.
- **PII redaction (`src/eval/redact.py`) is pattern-based, not a
  guarantee.** It catches structurally-recognizable categories (emails,
  phone numbers, card numbers, IPs, VIN-shaped strings), not every way
  sensitive data can appear in free text.
