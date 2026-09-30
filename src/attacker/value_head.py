import copy
import os

import torch
import torch.nn as nn

from src.model.transformer import GPT, GPTConfig


class GPTWithValueHead(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig | None = None,
        hf_model_name: str | None = None,
        torch_dtype=None,
        peft_config: dict | None = None,
        adapter_path: str | None = None,
        adapter_is_trainable: bool = True,
    ):
        super().__init__()
        if (cfg is None) == (hf_model_name is None):
            raise ValueError('pass exactly one of cfg (from-scratch) or hf_model_name (warm-start)')
        if hf_model_name is not None:
            from transformers import AutoModelForCausalLM
            self.backend = 'hf'
            self.uses_peft = False
            load_kwargs = {'low_cpu_mem_usage': True, 'attn_implementation': 'sdpa'}
            if torch_dtype is not None:
                load_kwargs['torch_dtype'] = torch_dtype
            base_model = AutoModelForCausalLM.from_pretrained(hf_model_name, **load_kwargs)
            if adapter_path is not None:
                from peft import PeftModel
                self.hf_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=adapter_is_trainable,
                )
                self.uses_peft = True
            elif peft_config is not None:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=peft_config.get('r', 16),
                    lora_alpha=peft_config.get('lora_alpha', 32),
                    lora_dropout=peft_config.get('lora_dropout', 0.05),
                    target_modules=peft_config.get('target_modules'),
                    bias='none',
                    task_type='CAUSAL_LM',
                )
                self.hf_model = get_peft_model(base_model, lora_config)
                self.uses_peft = True
            else:
                self.hf_model = base_model
            hidden_size = self.hf_model.config.hidden_size
            model_dtype = next(self.hf_model.parameters()).dtype
            self.value_head = nn.Linear(hidden_size, 1, bias=False, dtype=model_dtype)
        else:
            self.backend = 'scratch'
            self.uses_peft = False
            self.gpt = GPT(cfg)
            self.value_head = nn.Linear(cfg.n_embd, 1, bias=False)

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, attention_mask: torch.Tensor | None = None):
        if self.backend == 'hf':
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            if top_k is not None:
                gen_kwargs['top_k'] = top_k
            if attention_mask is not None:
                gen_kwargs['attention_mask'] = attention_mask
            return self.hf_model.generate(idx, **gen_kwargs)
        return self.gpt.generate(idx, max_new_tokens, temperature, top_k)

    def forward_logits_and_values(self, full_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = full_ids.unsqueeze(0) if full_ids.ndim == 1 else full_ids
        if self.backend == 'hf':
            out = self.hf_model(x, attention_mask=attention_mask, output_hidden_states=True)
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
        if self.backend == 'hf' and self.uses_peft:
            return ReferenceBackbone(self.hf_model, 'hf_peft_shared')
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
    def from_warm_start(cls, hf_model_name_or_path: str, device: str, peft_config: dict | None = None) -> 'GPTWithValueHead':
        if device.startswith('cuda'):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        model = cls(
            hf_model_name=hf_model_name_or_path,
            torch_dtype=dtype,
            peft_config=peft_config,
        ).to(device)
        value_head_path = os.path.join(hf_model_name_or_path, 'value_head.pt')
        if os.path.isdir(hf_model_name_or_path) and os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        return model

    @classmethod
    def from_peft_adapter(cls, adapter_path: str, device: str, is_trainable: bool = True) -> 'GPTWithValueHead':
        from peft import PeftConfig
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        if device.startswith('cuda'):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        model = cls(
            hf_model_name=adapter_config.base_model_name_or_path,
            torch_dtype=dtype,
            adapter_path=adapter_path,
            adapter_is_trainable=is_trainable,
        ).to(device)
        value_head_path = os.path.join(adapter_path, 'value_head.pt')
        if os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        return model


class ReferenceBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, backend: str):
        super().__init__()
        self.backbone = backbone
        self.backend = backend

    @torch.no_grad()
    def logits(self, full_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = full_ids.unsqueeze(0) if full_ids.ndim == 1 else full_ids
        if self.backend in {'hf_peft_shared', 'hf'}:
            was_training = self.backbone.training
            self.backbone.eval()
            try:
                if self.backend == 'hf_peft_shared':
                    with self.backbone.disable_adapter():
                        return self.backbone(x, attention_mask=attention_mask).logits
                return self.backbone(x, attention_mask=attention_mask).logits
            finally:
                self.backbone.train(was_training)
        g = self.backbone
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        h = g.dropout(g.tok_emb(x) + g.pos_emb(pos))
        for block in g.blocks:
            h = block(h)
        h = g.ln_f(h)
        return g.head(h)
