param(
    [switch]$SkipJudgeTraining
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python data/build_demo_assets.py
    if (-not $SkipJudgeTraining) {
        python -m src.judge.train_judge --train data/judge_train/train.jsonl --val data/judge_train/val.jsonl --out_dir checkpoints/judge_model --epochs 1 --batch_size 16 --max_len 128 --seed 42
        python -m src.judge.train_judge --train data/eval_judge_train/train.jsonl --val data/eval_judge_train/val.jsonl --out_dir checkpoints/eval_judge_model --epochs 1 --batch_size 16 --max_len 128 --seed 84
    }
    python -m src.attacker.rl_finetune --config configs/smoke_test.yaml
    python -m src.eval.run_campaign --config configs/smoke_test.yaml --seeds data/seed_prompts_smoke.txt --campaign_id smoke_e2e
}
finally {
    Pop-Location
}
