import concurrent.futures
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.targets.allowlist import CircuitBreaker
from src.targets.target_interface import TargetModel


class LocalTarget(TargetModel):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        device: str | None = None,
        dtype: str = 'auto',
        quantization: str | None = None,
        attn_implementation: str | None = 'sdpa',
        max_input_tokens: int = 2048,
        max_new_tokens: int = 128,
        batch_size: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        timeout_s: float = 60.0,
        circuit_breaker_threshold: int = 5,
    ):
        self.device = device if device and device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('target.device requests CUDA, but torch.cuda.is_available() is false')
        if max_input_tokens < 1 or max_new_tokens < 1 or batch_size < 1:
            raise ValueError('max_input_tokens, max_new_tokens, and batch_size must all be positive')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {'low_cpu_mem_usage': True}
        resolved_dtype = self._resolve_dtype(dtype)
        if resolved_dtype is not None:
            load_kwargs['torch_dtype'] = resolved_dtype
        if attn_implementation:
            load_kwargs['attn_implementation'] = attn_implementation
        if quantization:
            if not self.device.startswith('cuda'):
                raise ValueError('4bit/8bit quantization currently requires a CUDA target device')
            if quantization == '4bit':
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=resolved_dtype or torch.float16,
                    bnb_4bit_quant_type='nf4',
                )
            elif quantization == '8bit':
                load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("target.quantization must be null, '4bit', or '8bit'")
            load_kwargs['device_map'] = {'': self.device}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not quantization:
            self.model.to(self.device)
        self.model.eval()
        if self.device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True

        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.timeout_s = timeout_s
        self._breaker = CircuitBreaker(threshold=circuit_breaker_threshold)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _resolve_dtype(self, dtype: str):
        if dtype == 'auto':
            if not self.device.startswith('cuda'):
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        try:
            return dtype_map[dtype]
        except KeyError as exc:
            raise ValueError("target.dtype must be 'auto', 'float32', 'float16', or 'bfloat16'") from exc

    def respond(self, conversation: list[dict]) -> str:
        return self.respond_many([conversation])[0]

    def respond_many(self, conversations: list[list[dict]]) -> list[str]:
        if not conversations:
            return []
        self._breaker.before_call()
        try:
            responses = []
            for chunk in self._chunks(conversations, self.batch_size):
                generated = self._executor.submit(self._generate_many, chunk).result(timeout=self.timeout_s)
                responses.extend(generated)
        except concurrent.futures.TimeoutError:
            self._breaker.record_failure()
            raise TimeoutError(f'local target batch generation exceeded {self.timeout_s}s -- the campaign stopped waiting instead of hanging forever')
        except Exception:
            self._breaker.record_failure()
            raise
        else:
            self._breaker.record_success()
            return responses

    @staticmethod
    def _chunks(items: list, size: int) -> Iterable[list]:
        for start in range(0, len(items), size):
            yield items[start:start + size]

    def _render_conversation(self, conversation: list[dict]) -> str:
        kwargs = {'tokenize': False, 'add_generation_prompt': True}
        try:
            return self.tokenizer.apply_chat_template(conversation, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def _generate_many(self, conversations: list[list[dict]]) -> list[str]:
        prompts = [self._render_conversation(conversation) for conversation in conversations]
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        do_sample = self.temperature > 0
        generation_kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': do_sample,
            'use_cache': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'repetition_penalty': self.repetition_penalty,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        with torch.inference_mode():
            out = self.model.generate(**inputs, **generation_kwargs)
        input_width = inputs['input_ids'].shape[1]
        return [
            self.tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
            for row in out
        ]
