"""Model clients. One OpenAI-compatible client covers every open-weight setup:

  Ollama      base_url: http://localhost:11434/v1
  LM Studio   base_url: http://localhost:1234/v1
  vLLM        base_url: http://localhost:8000/v1
  OpenRouter  base_url: https://openrouter.ai/api/v1         (api_key_env: OPENROUTER_API_KEY)
  HF router   base_url: https://router.huggingface.co/v1     (api_key_env: HF_TOKEN)

Plus a `mock` provider that needs no model at all, for testing the pipeline.
Model settings live in config/models/<name>.yaml.
"""
from __future__ import annotations

import base64
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from hotel_lab import CONFIG_DIR

MODELS_DIR = CONFIG_DIR / "models"


@dataclass
class ModelConfig:
    name: str
    provider: str                       # openai_compatible | mock
    model: str = ""
    base_url: str = ""
    api_key_env: Optional[str] = None
    family: str = ""
    size_b: Optional[float] = None
    quantisation: str = ""
    serving: str = ""                   # ollama | lmstudio | vllm | openrouter | hf | mock
    temperature: float = 1.0
    max_tokens: int = 400
    supports_vision: bool = False
    json_mode: bool = False             # ask the server for JSON output (if it supports it)
    send_seed: bool = True              # pass the trial seed to the server (Ollama/vLLM honour it)
    timeout_s: float = 120.0
    strategy: str = "random"            # mock only: random | first | biased_first
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "ModelConfig":
        p = Path(name_or_path)
        if not p.exists():
            p = MODELS_DIR / f"{name_or_path}.yaml"
        if not p.exists():
            avail = ", ".join(sorted(x.stem for x in MODELS_DIR.glob("*.yaml")))
            raise FileNotFoundError(f"No model config '{name_or_path}'. Available: {avail}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.setdefault("name", p.stem)
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{p.name}: unknown fields {sorted(unknown)}")
        return cls(**data)


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_s: float = 0.0


def image_part(png_bytes: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(png_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._client = None
        if cfg.provider == "openai_compatible":
            from openai import OpenAI
            key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
            if cfg.api_key_env and not key:
                raise RuntimeError(f"{cfg.name}: environment variable {cfg.api_key_env} is not set (put it in .env)")
            self._client = OpenAI(base_url=cfg.base_url, api_key=key or "not-needed", timeout=cfg.timeout_s)
        elif cfg.provider != "mock":
            raise ValueError(f"{cfg.name}: unknown provider '{cfg.provider}'")

    def complete(self, prompt: str, images: Optional[list[bytes]] = None, seed: Optional[int] = None,
                 n_options: int = 4) -> Completion:
        t0 = time.time()
        if self.cfg.provider == "mock":
            return Completion(self._mock(seed, n_options), latency_s=time.time() - t0)

        if images:
            if not self.cfg.supports_vision:
                raise RuntimeError(f"{self.cfg.name} is not marked supports_vision: true")
            content: Any = [{"type": "text", "text": prompt}] + [image_part(b) for b in images]
        else:
            content = prompt
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if self.cfg.send_seed and seed is not None:
            kwargs["seed"] = seed
        if self.cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return Completion(
            text=(resp.choices[0].message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=time.time() - t0,
        )

    def _mock(self, seed: Optional[int], n: int) -> str:
        rng = random.Random(seed)
        if self.cfg.strategy == "first":
            pick = 1
        elif self.cfg.strategy == "biased_first":   # strong position-1 bias, for testing the analysis
            pick = 1 if rng.random() < 0.5 else rng.randint(1, n)
        else:
            pick = rng.randint(1, n)
        return json.dumps({"option_number": pick, "hotel_name": "", "reason": f"mock ({self.cfg.strategy})"})

    def describe_version(self) -> dict[str, Any]:
        """Best-effort model fingerprint for the log (detects silent model updates = drift)."""
        info: dict[str, Any] = {}
        if self.cfg.serving == "ollama" and self.cfg.base_url:
            try:
                import httpx
                root = self.cfg.base_url.rstrip("/").removesuffix("/v1")
                tags = httpx.get(f"{root}/api/tags", timeout=5).json().get("models", [])
                for m in tags:
                    if m.get("name") == self.cfg.model or m.get("model") == self.cfg.model:
                        info["digest"] = m.get("digest")
                        info["quantization_level"] = (m.get("details") or {}).get("quantization_level")
                        info["parameter_size"] = (m.get("details") or {}).get("parameter_size")
            except Exception as e:  # server down, older Ollama, etc.
                info["version_error"] = str(e)
        return info
