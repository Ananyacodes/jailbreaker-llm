import copy
import os

import torch
import torch.nn as nn

from src.model.transformer import GPT, GPTConfig


class GPTWithValueHead(nn.Module):
    def __init__(self, cfg: GPTConfig | None = None, hf_model_name: str | None = None):
        super().__init__()
        if (cfg is None) == (hf_model_name is None):
            raise ValueError('pass exactly one of cfg (from-scratch) or hf_model_name (warm-start)')
        if hf_model_name is not None:
            from transformers import AutoModelForCausalLM
            self.backend = 'hf'
            self.hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
            hidden_size = self.hf_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.backend = 'scratch'
            self.gpt = GPT(cfg)
            self.value_head = nn.Linear(cfg.n_embd, 1, bias=False)

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        if self.backend == 'hf':
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            if top_k is not None:
                gen_kwargs['top_k'] = top_k
            return self.hf_model.generate(idx, **gen_kwargs)
        return self.gpt.generate(idx, max_new_tokens, temperature, top_k)

    def forward_logits_and_values(self, full_ids: torch.Tensor):
        x = full_ids.unsqueeze(0)
        if self.backend == 'hf':
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
        return (logits, values)

    def make_reference(self, device: str) -> 'ReferenceBackbone':
        if self.backend == 'hf':
            ref = copy.deepcopy(self.hf_model).to(device).eval()
        else:
            ref = copy.deepcopy(self.gpt).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ReferenceBackbone(ref, self.backend)

    def save_pretrained(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.backend == 'hf':
            self.hf_model.save_pretrained(out_dir)
            torch.save(self.value_head.state_dict(), f'{out_dir}/value_head.pt')
        else:
            torch.save({'model': self.gpt.state_dict(), 'value_head': self.value_head.state_dict(), 'config': self.gpt.cfg}, f'{out_dir}/latest.pt')

    @classmethod
    def from_pretrained_gpt(cls, ckpt_path: str, device: str) -> 'GPTWithValueHead':
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls(cfg=ckpt['config']).to(device)
        model.gpt.load_state_dict(ckpt['model'])
        return model

    @classmethod
    def from_warm_start(cls, hf_model_name_or_path: str, device: str) -> 'GPTWithValueHead':
        model = cls(hf_model_name=hf_model_name_or_path).to(device)
        value_head_path = os.path.join(hf_model_name_or_path, 'value_head.pt')
        if os.path.isdir(hf_model_name_or_path) and os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        return model


class ReferenceBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, backend: str):
        super().__init__()
        self.backbone = backbone
        self.backend = backend

    @torch.no_grad()
    def logits(self, full_ids: torch.Tensor) -> torch.Tensor:
        x = full_ids.unsqueeze(0)
        if self.backend == 'hf':
            return self.backbone(x).logits
        g = self.backbone
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        h = g.dropout(g.tok_emb(x) + g.pos_emb(pos))
        for block in g.blocks:
            h = block(h)
        h = g.ln_f(h)
        return g.head(h)
