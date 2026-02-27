"""HuggingFace inference backend for SentryBench.

Supports:
- Plain base model (``adapter_path=None``)
- Base model + LoRA/QLoRA adapter (``adapter_path`` set)
- 4-bit / 8-bit quantisation via ``load_in_4bit`` / ``load_in_8bit``

The model is loaded lazily on first call to :meth:`generate` to avoid
blocking the pipeline when the model isn't needed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import torch

from .base import BaseModel
from ..registry import registry


class HFModel(BaseModel):
    """HuggingFace causal-LM inference backend.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace hub name or local path to the base model.
    adapter_path:
        Optional path to a PEFT LoRA adapter directory.  If supplied,
        the adapter is loaded on top of the base model via ``peft``.
    load_in_4bit:
        Load base model in 4-bit (bitsandbytes QLoRA-style).
    load_in_8bit:
        Load base model in 8-bit (bitsandbytes).
    device_map:
        Device map passed to ``from_pretrained``.  Defaults to ``"auto"``.
    max_new_tokens:
        Maximum tokens to generate per call.
    torch_dtype:
        Torch dtype string (``"float16"``, ``"bfloat16"``, ``"float32"``).
        Defaults to ``"auto"``.
    trust_remote_code:
        Whether to trust remote code in model configs.
    """

    name = "hf"

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model and tokenizer (called once, thread-safe)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.torch_dtype, "auto")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quant_cfg,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        if self.adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
            model = model.merge_and_unload()  # fuse adapter weights for faster inference

        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Batched generation with left-padding for efficiency."""
        self._ensure_loaded()
        tokenizer = self._tokenizer
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        tokenizer.padding_side = orig_padding_side

        results = []
        input_len = inputs["input_ids"].shape[-1]
        for ids in output_ids:
            new_ids = ids[input_len:]
            results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        return results


registry.register("model", HFModel.name, HFModel)
