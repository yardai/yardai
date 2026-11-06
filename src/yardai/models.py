"""Candidate open-weight base models for YardAI.

The plan is to start small and climb a ladder: prove low-communication training works at
~100M parameters, then scale. Licences must allow open redistribution of derived weights.
Verify each model card before use — licences can change between releases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModel:
    hf_id: str
    params: str
    licence: str
    why: str


LADDER: list[BaseModel] = [
    BaseModel(
        "HuggingFaceTB/SmolLM2-135M",
        "135M",
        "Apache-2.0",
        "Tiny, modern Llama-style architecture; fits on almost any device. Stage 1.",
    ),
    BaseModel(
        "HuggingFaceTB/SmolLM2-360M",
        "360M",
        "Apache-2.0",
        "Same family one step up; tests scaling of the protocol. Stage 2.",
    ),
    BaseModel(
        "Qwen/Qwen2.5-0.5B", "0.5B", "Apache-2.0", "Strong small multilingual base. Stage 3."
    ),
    BaseModel(
        "Qwen/Qwen2.5-1.5B",
        "1.5B",
        "Apache-2.0",
        "First model that needs model sharding on consumer devices. Stage 4.",
    ),
]


def load(hf_id: str = LADDER[0].hf_id):
    """Load a base model and tokenizer via Hugging Face (requires ``pip install yardai[hf]``)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the 'hf' extra: pip install 'yardai[hf]'") from e
    return AutoModelForCausalLM.from_pretrained(hf_id), AutoTokenizer.from_pretrained(hf_id)
