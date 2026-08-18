"""Thin inference wrapper for the train-only Reward Judge. Scores exactly
one (prompt, response) pair -- fast enough to call thousands of times
during PPO. Never used to score a final report; see eval_judge.py for that."""
import torch
from transformers import AutoTokenizer

from src.judge.train_judge import LABELS, JudgeModel


class RewardJudge:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = JudgeModel().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> dict:
        text = f"PROMPT: {prompt}\nRESPONSE: {response}"
        enc = self.tokenizer(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
        ).to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return {label: p for label, p in zip(LABELS, probs)}
