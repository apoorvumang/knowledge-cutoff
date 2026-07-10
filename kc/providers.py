"""Provider abstraction: call any registered model with a single interface.

Two backends:
  - anthropic : native Messages API
  - openai    : any OpenAI-compatible /chat/completions endpoint

`load_registry()` reads models.yaml. `get_model(key)` returns a `Model` you can
call with `.complete(system, user)`.

Design notes for a knowledge-cutoff benchmark:
  - We keep the system prompt EMPTY by default and never inject today's date,
    so the model can't infer recency from the harness.
  - temperature=0 for determinism.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

DEFAULT_MAX_TOKENS = 2048


@dataclass
class ModelSpec:
    key: str
    provider: str
    provider_kind: str
    model_id: str
    base_url: str | None
    api_key_env: str


class Registry:
    def __init__(self, providers: dict[str, Any], models: dict[str, ModelSpec]):
        self.providers = providers
        self.models = models

    def keys(self) -> list[str]:
        return sorted(self.models)


def load_registry(path: str = "models.yaml") -> Registry:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    providers = cfg["providers"]
    models: dict[str, ModelSpec] = {}
    for key, m in cfg["models"].items():
        pname = m["provider"]
        p = providers[pname]
        models[key] = ModelSpec(
            key=key,
            provider=pname,
            provider_kind=p["kind"],
            model_id=m["model_id"],
            base_url=p.get("base_url"),
            api_key_env=p["api_key_env"],
        )
    return Registry(providers, models)


class Model:
    """A callable model. `complete` returns (text, meta) or raises."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"env var {spec.api_key_env} is not set (needed for model {spec.key})")
        self._api_key = api_key
        self._client = None  # lazily constructed

    @property
    def key(self) -> str:
        return self.spec.key

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.spec.provider_kind == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif self.spec.provider_kind == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key, base_url=self.spec.base_url)
        else:
            raise RuntimeError(f"unknown provider kind {self.spec.provider_kind!r}")

    def complete(self, user: str, system: str = "", *,
                 temperature: float = 0.0,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[str, dict]:
        self._ensure_client()
        if self.spec.provider_kind == "anthropic":
            return self._complete_anthropic(user, system, temperature, max_tokens)
        return self._complete_openai(user, system, temperature, max_tokens)

    def _complete_anthropic(self, user, system, temperature, max_tokens):
        kwargs: dict[str, Any] = dict(
            model=self.spec.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        if system:
            kwargs["system"] = system
        # Some reasoning models reject temperature != 1; tolerate that.
        try:
            resp = self._client.messages.create(temperature=temperature, **kwargs)
        except Exception:
            resp = self._client.messages.create(**kwargs)
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text.strip(), {"stop_reason": resp.stop_reason}

    def _complete_openai(self, user, system, temperature, max_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = dict(model=self.spec.model_id, messages=messages)
        # max_completion_tokens is the newer name; fall back to max_tokens.
        try:
            resp = self._client.chat.completions.create(
                temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        except Exception:
            try:
                resp = self._client.chat.completions.create(
                    max_completion_tokens=max_tokens, **kwargs)
            except Exception:
                resp = self._client.chat.completions.create(
                    temperature=temperature, max_tokens=max_tokens, **kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        return text.strip(), {"finish_reason": choice.finish_reason}


def get_model(key: str, registry: Registry | None = None) -> Model:
    registry = registry or load_registry()
    if key not in registry.models:
        raise KeyError(
            f"unknown model {key!r}. known: {', '.join(registry.keys())}")
    return Model(registry.models[key])
