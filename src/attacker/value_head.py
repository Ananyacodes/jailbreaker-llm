"""
Adds a scalar value head on top of a causal LM backbone for PPO-style RL,
without ever needing a from-scratch checkpoint to go through an HF-format
conversion step.

Two backends, chosen once at construction and never mixed:

- "scratch": wraps our own src/model/transformer.py GPT class, loaded from
  a torch.save() checkpoint produced by src/model/pretrain.py's
  from-scratch loop. This is the original path -- see from_pretrained_gpt.
- "hf": wraps an existing open-weight causal LM straight from the
  HuggingFace hub/local dir (see configs/base_model.yaml:
  pretrain.warm_start_from). No conversion happens because none is needed
  -- the weights are already in their native format. This is the
  recommended default (see README "closing the gap with existing tools"):
  it skips weeks of from-scratch pretraining and gets to a first RL step
  immediately.

src/attacker/ppo_trainer.py only ever calls the methods below -- it never
reaches into either backbone's internals directly, so adding a third
backend later doesn't require touching the PPO loop.
"""
import copy
import os

import torch
import torch.nn as nn

from src.model.transformer import GPT, GPTConfig


class GPTWithValueHead(nn.Module):
    def __init__(self, cfg: GPTConfig | None = None, hf_model_name: str | None = None):
        super().__init__()
        if (cfg is None) == (hf_model_name is None):
            raise ValueError("pass exactly one of cfg (from-scratch) or hf_model_name (warm-start)")

        if hf_model_name is not None:
            from transformers import AutoModelForCausalLM

            self.backend = "hf"
            self.hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
            hidden_size = self.hf_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.backend = "scratch"
            self.gpt = GPT(cfg)
            self.value_head = nn.Linear(cfg.n_embd, 1, bias=False)

    # ---- generation ----

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        if self.backend == "hf":
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            return self.hf_model.generate(idx, **gen_kwargs)
        return self.gpt.generate(idx, max_new_tokens, temperature, top_k)

    # ---- teacher-forced logits + values, single sequence (batch dim 1) ----
    # Used by SimplePPOTrainer._logprobs_and_values -- keeping this as the
    # one seam between the two backends means the PPO loop itself never
    # branches on self.backend.

    def forward_logits_and_values(self, full_ids: torch.Tensor):
        """full_ids: (T,) token ids, no batch dim. Returns (logits, values),
        both (1, T, ...) -- caller handles the shift-by-one for targets."""
        x = full_ids.unsqueeze(0)
        if self.backend == "hf":
            out = self.hf_model(x, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            logits = out.logits
        else:
            g = self.gpt
            B, T = x.shape
            pos = torch.arange(0, T, device=x.device)
            h = g.dropout(g.tok_emb(x) + g.pos_emb(pos))
            for block in g.blocks:
                h = block(h)
            hidden = g.ln_f(h)
            logits = g.head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    # ---- frozen reference copy for the KL penalty ----

    def make_reference(self, device: str) -> "ReferenceBackbone":
        if self.backend == "hf":
            ref = copy.deepcopy(self.hf_model).to(device).eval()
        else:
            ref = copy.deepcopy(self.gpt).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ReferenceBackbone(ref, self.backend)

    # ---- persistence ----

    def save_pretrained(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.backend == "hf":
            self.hf_model.save_pretrained(out_dir)
            torch.save(self.value_head.state_dict(), f"{out_dir}/value_head.pt")
        else:
            torch.save(
                {"model": self.gpt.state_dict(), "value_head": self.value_head.state_dict(), "config": self.gpt.cfg},
                f"{out_dir}/latest.pt",
            )

    @classmethod
    def from_pretrained_gpt(cls, ckpt_path: str, device: str) -> "GPTWithValueHead":
        """Loads a from-scratch checkpoint produced by src/model/pretrain.py.
        weights_only=False: this only ever loads checkpoints this pipeline
        produced itself -- never a third-party/untrusted file."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls(cfg=ckpt["config"]).to(device)
        model.gpt.load_state_dict(ckpt["model"])
        return model

    @classmethod
    def from_warm_start(cls, hf_model_name_or_path: str, device: str) -> "GPTWithValueHead":
        """Loads an existing open-weight causal LM directly -- see the
        module docstring for why this is the recommended default.

        If hf_model_name_or_path is a local directory that also contains a
        value_head.pt (i.e. this is reloading a checkpoint save_pretrained
        wrote earlier -- resuming RL fine-tuning, or loading a trained
        attacker for a campaign), that value head is restored too. A fresh
        first-time warm start from a hub id has no such file and gets a
        newly-initialized value head, same as the from-scratch path."""
        model = cls(hf_model_name=hf_model_name_or_path).to(device)
        value_head_path = os.path.join(hf_model_name_or_path, "value_head.pt")
        if os.path.isdir(hf_model_name_or_path) and os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        return model


class ReferenceBackbone(nn.Module):
    """Frozen copy of just the backbone (no value head) used for the
    KL-to-reference penalty in ppo_trainer.py -- prevents the attacker
    from drifting into reward-hacking gibberish."""

    def __init__(self, backbone: nn.Module, backend: str):
        super().__init__()
        self.backbone = backbone
        self.backend = backend

    @torch.no_grad()
    def logits(self, full_ids: torch.Tensor) -> torch.Tensor:
        x = full_ids.unsqueeze(0)
        if self.backend == "hf":
            return self.backbone(x).logits
        g = self.backbone
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        h = g.dropout(g.tok_emb(x) + g.pos_emb(pos))
        for block in g.blocks:
            h = block(h)
        h = g.ln_f(h)
        return g.head(h)
