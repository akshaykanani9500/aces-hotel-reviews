"""Model clients. One OpenAI-compatible client covers every open-weight setup:

  Ollama      base_url: http://localhost:11434/v1
  LM Studio   base_url: http://localhost:1234/v1
  vLLM        base_url: http://localhost:8000/v1
  OpenRouter  base_url: https://openrouter.ai/api/v1         (api_key_env: OPENROUTER_API_KEY)
  HF router   base_url: https://router.huggingface.co/v1     (api_key_env: HF_TOKEN)

For Ollama we use its NATIVE API (provider: ollama) instead of /v1, because only the native API lets us
set the context window (num_ctx). Ollama's default window (2048-4096 tokens) is smaller than our
full-review prompts (~5k tokens) and Ollama cuts the START of a long prompt without any error, which
would silently delete the instructions and Option 1.

Closed models (only if your professor allows them):
  OpenAI      provider: openai_compatible, base_url: https://api.openai.com/v1   (OPENAI_API_KEY)
  Gemini      provider: openai_compatible, base_url: https://generativelanguage.googleapis.com/v1beta/openai/  (GEMINI_API_KEY)
  Anthropic   provider: anthropic (native Messages API)                          (ANTHROPIC_API_KEY)

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
# Extra no-AI agents used only by the automated tests and `demo`; the Studio does not list them.
TEST_MODELS_DIR = Path(__file__).resolve().parent / "tests" / "models"
TEST_AGENT_PREFIXES = ("mock", "no-ai")


def is_test_agent(name: str, provider_or_serving: str = "") -> bool:
    """True for agents that are not a real AI (never pooled into real results)."""
    return provider_or_serving == "mock" or str(name).startswith(TEST_AGENT_PREFIXES)


@dataclass
class ModelConfig:
    name: str
    provider: str                       # ollama | openai_compatible | anthropic | mock
    model: str = ""
    base_url: str = ""
    api_key_env: Optional[str] = None
    family: str = ""
    size_b: Optional[float] = None
    quantisation: str = ""
    serving: str = ""                   # ollama | lmstudio | vllm | openrouter | hf | mock
    temperature: float = 1.0
    max_tokens: int = 400
    num_ctx: int = 16384                # context window (Ollama); prompt + reply must fit
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
            p = TEST_MODELS_DIR / f"{name_or_path}.yaml"
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
    finish_reason: Optional[str] = None      # "stop" is good; "length" means the reply was cut off
    response_meta: dict[str, Any] = field(default_factory=dict)  # served model id, provider, fingerprint


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
        elif cfg.provider == "ollama":
            import httpx
            self._http = httpx.Client(base_url=cfg.base_url.rstrip("/").removesuffix("/v1"), timeout=cfg.timeout_s)
        elif cfg.provider == "anthropic":
            import httpx
            key = os.getenv(cfg.api_key_env or "ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(f"{cfg.name}: environment variable {cfg.api_key_env or 'ANTHROPIC_API_KEY'} is not set")
            self._http = httpx.Client(base_url=(cfg.base_url or "https://api.anthropic.com").rstrip("/"),
                                      timeout=cfg.timeout_s,
                                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        elif cfg.provider != "mock":
            raise ValueError(f"{cfg.name}: unknown provider '{cfg.provider}'")

    def complete(self, prompt: str, images: Optional[list[bytes]] = None, seed: Optional[int] = None,
                 n_options: int = 4) -> Completion:
        t0 = time.time()
        if self.cfg.provider == "mock":
            return Completion(self._mock(seed, n_options, prompt), latency_s=time.time() - t0)

        if images and not self.cfg.supports_vision:
            raise RuntimeError(f"{self.cfg.name} is not marked supports_vision: true")
        if self.cfg.provider == "ollama":
            return self._ollama(prompt, images, seed, t0)
        if self.cfg.provider == "anthropic":
            return self._anthropic(prompt, images, t0)
        if images:
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
        resp = self._create_with_fallbacks(kwargs)
        usage = getattr(resp, "usage", None)
        meta = {"served_model": getattr(resp, "model", None), "response_id": getattr(resp, "id", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None)}
        provider = getattr(resp, "provider", None) or (getattr(resp, "model_extra", None) or {}).get("provider")
        if provider:
            meta["provider"] = provider  # OpenRouter: which upstream provider actually served the call
        return Completion(
            text=(resp.choices[0].message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=time.time() - t0,
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
            response_meta=meta,
        )

    def _create_with_fallbacks(self, kwargs: dict[str, Any]):
        """Some providers reject parameters (newer OpenAI models want max_completion_tokens and only the
        default temperature; some servers reject seed or response_format). Retry without them and
        remember what was dropped, so it is logged."""
        from openai import BadRequestError
        self.dropped_params: list[str] = getattr(self, "dropped_params", [])
        for _ in range(4):
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e).lower()
                if "max_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    self.dropped_params.append("max_tokens->max_completion_tokens")
                elif "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    self.dropped_params.append("temperature (provider default used)")
                elif "seed" in msg and "seed" in kwargs:
                    kwargs.pop("seed")
                    self.dropped_params.append("seed")
                elif "response_format" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format")
                    self.dropped_params.append("response_format")
                else:
                    raise
        return self._client.chat.completions.create(**kwargs)

    def _anthropic(self, prompt: str, images: Optional[list[bytes]], t0: float) -> Completion:
        content: list[dict[str, Any]] = []
        for b in images or []:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.b64encode(b).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "temperature": self.cfg.temperature,
                "messages": [{"role": "user", "content": content}]}
        r = self._http.post("/v1/messages", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Anthropic error {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(text=text, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                          latency_s=time.time() - t0, finish_reason=data.get("stop_reason"),
                          response_meta={"served_model": data.get("model"), "response_id": data.get("id")})

    def _ollama(self, prompt: str, images: Optional[list[bytes]], seed: Optional[int], t0: float) -> Completion:
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        options: dict[str, Any] = {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx,
                                   "num_predict": self.cfg.max_tokens}
        if self.cfg.send_seed and seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": [msg], "stream": False, "options": options}
        if self.cfg.json_mode:
            payload["format"] = "json"
        payload.update(self.cfg.extra_body or {})   # e.g. {"think": false} for thinking models
        r = self._http.post("/api/chat", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return Completion(
            text=(data.get("message") or {}).get("content", "") or "",
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_s=time.time() - t0,
            finish_reason=data.get("done_reason"),
            response_meta={"served_model": data.get("model")},
        )

    def _mock(self, seed: Optional[int], n: int, prompt: str = "") -> str:
        rng = random.Random(seed)
        if self.cfg.strategy == "simulated":
            return self._simulated(rng, n, prompt)
        if self.cfg.strategy == "first":
            pick = 1
        elif self.cfg.strategy == "biased_first":   # strong position-1 bias, for testing the analysis
            pick = 1 if rng.random() < 0.5 else rng.randint(1, n)
        else:
            pick = rng.randint(1, n)
        return json.dumps({"option_number": pick, "hotel_name": "", "reason": f"mock ({self.cfg.strategy})"})

    def _simulated(self, rng: random.Random, n: int, prompt: str) -> str:
        """A fake 'agent' with KNOWN preferences, to check that the analysis recovers them.
        Utility per room = w_rating * mean shown stars + w_emoji * (has emoji) + w_long * (long reviews)
        + w_first * (first review stars - 4) + w_position1 * (is option 1) + Gumbel noise.
        Weights come from `extra_body` in the model YAML. Text interface only."""
        import math
        import re
        w = {"rating": 2.0, "emoji": 0.0, "long": 0.0, "first": 0.0, "position1": 0.0, "facts": 2.0}
        w.update({k: float(v) for k, v in (self.cfg.extra_body or {}).items()})
        try:
            start = prompt.index("[")
            rooms = json.loads(prompt[start:prompt.rindex("]") + 1])
        except Exception:
            return json.dumps({"option_number": rng.randint(1, n), "reason": "test agent is working (no rooms in this prompt)"})
        emoji_re = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
        best, pick = -1e9, 1
        for r in rooms:
            revs = r.get("reviews") or []
            u = 0.0
            if revs:
                u += w["rating"] * (sum(x["rating"] for x in revs) / len(revs) - 4)
                u -= 0.6 * w["rating"] * sum(1 for x in revs if x["rating"] <= 2)       # negativity
                u += w["emoji"] * (1.0 if any(emoji_re.search(x["text"]) for x in revs) else 0.0)
                u += w["long"] * (1.0 if sum(len(x["text"].split()) for x in revs) / len(revs) > 30 else 0.0)
                u += w["first"] * (revs[0]["rating"] - 4)
                bad = sum(len(re.findall(r"\b(slow|noisy|noise|dirty|cold|rude|broken|poor|drops?|weak|stain|dust|lukewarm|late|lumpy|freezing|sagging|thin|loud|honking|bland|disappointing|letdown|unhelpful|cramped|hard|damp|smell)\b",
                                          x["text"].lower())) for x in revs)
                u -= w["facts"] * bad / len(revs)                 # 'reads' complaints in the text
            for line in r.get("verified_facts") or []:
                u += {"Good": 1.5, "Average": 0.0, "Poor": -1.5}.get(line.split(": ")[-1], 0.0) * w["facts"]
            if r.get("review_summary"):
                s = r["review_summary"].lower()
                u -= w["facts"] * len(re.findall(r"\b(slow|noisy|noise|dirty|cold|rude|broken|poor|drops?|weak|stain|dust|lukewarm|late|lumpy|freezing|sagging|thin|loud|honking|bland|disappointing|letdown|unhelpful|cramped|hard|damp|smell)\b", s)) / 2
            u += w["position1"] * (1.0 if r.get("option_number") == 1 else 0.0)
            u += -math.log(-math.log(max(rng.random(), 1e-12)))
            if u > best:
                best, pick = u, r.get("option_number", 1)
        return json.dumps({"option_number": pick, "hotel_name": "", "reason": "simulated preferences"})

    def describe_version(self) -> dict[str, Any]:
        """Best-effort model fingerprint for the log (detects silent model updates = drift)."""
        info: dict[str, Any] = {}
        if (self.cfg.provider == "ollama" or self.cfg.serving == "ollama") and self.cfg.base_url:
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
