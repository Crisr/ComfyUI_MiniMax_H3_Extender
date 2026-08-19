"""MiniMax H3 external prompt bridge.

Packs numbered STRING prompt outputs from story/prompt generator nodes into one
H3_PROMPT_PACK socket that the MiniMax H3 Extender can import into its normal
per-clip prompt cards.
"""

from __future__ import annotations

import hashlib
import json

MAX_PROMPTS = 10
PROMPT_PACK_TYPE = "H3_PROMPT_PACK"
PROMPT_PACK_VERSION = 1


def _pack_prompts(values):
    """Return one contiguous prompt list and reject accidental holes.

    H3 Story -> Sequences exposes ten fixed STRING outputs and returns empty
    strings after n_sequences. Keeping the list contiguous prevents a missing
    middle cable/output from silently shifting prompt numbering.
    """
    prompts = []
    found_empty = False
    for index, value in enumerate(list(values)[:MAX_PROMPTS], start=1):
        text = "" if value is None else str(value)
        if not text.strip():
            found_empty = True
            continue
        if found_empty:
            raise ValueError(
                "MiniMax H3 Prompt Pack Bridge: prompt outputs must be contiguous "
                f"from prompt_1. Found non-empty prompt_{index} after an empty slot."
            )
        prompts.append(text)

    if not prompts:
        raise ValueError(
            "MiniMax H3 Prompt Pack Bridge: no non-empty prompts were received."
        )
    return prompts


def _prompt_pack_signature(prompts):
    payload = json.dumps(
        list(prompts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MiniMaxH3PromptPackBridge:
    """Pack prompt_1..prompt_10 STRING outputs into a single Extender input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect prompt_1 from an external H3 story/prompt node.",
                    },
                ),
            },
            "optional": {
                **{
                    f"prompt_{i}": (
                        "STRING",
                        {
                            "forceInput": True,
                            "tooltip": f"Connect external prompt_{i}. Trailing empty prompts are ignored.",
                        },
                    )
                    for i in range(2, MAX_PROMPTS + 1)
                },
            },
        }

    RETURN_TYPES = (PROMPT_PACK_TYPE, "INT")
    RETURN_NAMES = ("prompt_pack", "prompt_count")
    FUNCTION = "pack"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = False

    def pack(self, prompt_1, **kwargs):
        values = [prompt_1]
        values.extend(kwargs.get(f"prompt_{i}") for i in range(2, MAX_PROMPTS + 1))
        prompts = _pack_prompts(values)
        signature = _prompt_pack_signature(prompts)
        pack = {
            "type": PROMPT_PACK_TYPE,
            "version": PROMPT_PACK_VERSION,
            "source": "MiniMax H3 Prompt Pack Bridge",
            "count": len(prompts),
            "prompts": prompts,
            "signature": signature,
        }
        return (pack, int(len(prompts)))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptPackBridge": MiniMaxH3PromptPackBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptPackBridge": "MiniMax H3 Prompt Pack Bridge",
}
