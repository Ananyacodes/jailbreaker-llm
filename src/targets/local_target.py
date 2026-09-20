import concurrent.futures

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.targets.allowlist import CircuitBreaker
from src.targets.target_interface import TargetModel


class LocalTarget(TargetModel):
    def __init__(self, model_name: str, device: str | None = None, max_new_tokens: int = 256, timeout_s: float = 60.0, circuit_breaker_threshold: int = 5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self._breaker = CircuitBreaker(threshold=circuit_breaker_threshold)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def respond(self, conversation: list[dict]) -> str:
        self._breaker.before_call()
        try:
            text = self._executor.submit(self._generate, conversation).result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            self._breaker.record_failure()
            raise TimeoutError(f'local target generation exceeded {self.timeout_s}s -- a single hung generation should not stall the whole campaign, so this call gives up rather than blocking forever')
        except Exception:
            self._breaker.record_failure()
            raise
        else:
            self._breaker.record_success()
            return text

    def _generate(self, conversation: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7)
        text = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return text.strip()
