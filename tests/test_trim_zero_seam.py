"""Regression tests: a trim-0 seam must decode solo, never as a pair.

A trim-0 seam is what a per-clip Motion Context OFF clip leaves behind: the
continuation duplicates no frames, so the join is a hard cut. The pair decoder
cannot express that. It concatenates ``prev`` and ``next`` latents and requires
``(prev_T - decode_start_t) % 5 == 0``; H3 video latents always satisfy
``prev_T % 5 == 2`` and a trim-0 seam has ``decode_start_t == 0``, so every
trim-0 pair is rejected with "H3 temporal phase mismatch".

These tests pin the dispatch, without importing ComfyUI.
"""

import ast
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parent.parent
DISK = ROOT / "motion_context_disk.py"


def load_functions(*names, **extra_globals):
    """Execute the named top-level functions from motion_context_disk.py."""
    tree = ast.parse(DISK.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in nodes}
    missing = set(names) - found
    if missing:
        raise AssertionError(f"functions not found: {sorted(missing)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"int": int, "torch": torch}
    namespace.update(extra_globals)
    exec(compile(module, str(DISK), "exec"), namespace)
    return namespace


class PairUsed(Exception):
    """Raised by a stub to prove the pair decoder was reached."""


class VideoSeamDispatchTest(unittest.TestCase):
    def setUp(self):
        self.loaded = []

        def load_segment_video(data_path, desc):
            self.loaded.append(desc)
            return torch.zeros(1, 4, 17, 2, 2)

        def build_pair_video(*args, **kwargs):
            raise PairUsed("pair decoder used")

        def decode_pair_video(*args, **kwargs):
            raise PairUsed("pair decoder used")

        self.ns = load_functions(
            "_render_one_final_video_segment",
            _load_segment_video=load_segment_video,
            _build_pair_video=build_pair_video,
            _decode_pair_video=decode_pair_video,
            _correct_current_segment=lambda prev, curr: curr,
        )

    def test_trim_zero_continuation_decodes_solo(self):
        segments = [{"frames": 122}, {"frames": 97, "trim_frames": 0}]

        class VAE:
            def decode(self, latent):
                return torch.zeros(97, 3, 8, 8)

        video, shift = self.ns["_render_one_final_video_segment"](
            "unused", segments, 1, VAE()
        )

        self.assertEqual(shift, 0)
        self.assertEqual(int(video.shape[0]), 97)
        self.assertEqual(self.loaded, [segments[1]])

    def test_positive_trim_still_uses_pair_decoder(self):
        segments = [{"frames": 122}, {"frames": 97, "trim_frames": 5}]

        with self.assertRaises(PairUsed):
            self.ns["_render_one_final_video_segment"](
                "unused", segments, 1, object()
            )


class AudioSeamDispatchTest(unittest.TestCase):
    def setUp(self):
        self.solo = []

        def decode_single_audio(data_path, desc, audio_vae, fps):
            self.solo.append(desc)
            return {"waveform": torch.zeros(1, 2, 10), "sample_rate": 40}

        def decode_pair_audio(*args, **kwargs):
            raise PairUsed("pair decoder used")

        self.ns = load_functions(
            "_render_one_final_audio_segment",
            _decode_single_audio=decode_single_audio,
            _decode_pair_audio=decode_pair_audio,
        )

    def test_trim_zero_continuation_decodes_solo(self):
        segments = [{"frames": 122}, {"frames": 97, "trim_frames": 0}]

        audio = self.ns["_render_one_final_audio_segment"](
            "unused", segments, 1, object(), 24.0, 0
        )

        self.assertEqual(self.solo, [segments[1]])
        self.assertEqual(int(audio["sample_rate"]), 40)

    def test_positive_trim_still_uses_pair_decoder(self):
        segments = [{"frames": 122}, {"frames": 97, "trim_frames": 5}]

        with self.assertRaises(PairUsed):
            self.ns["_render_one_final_audio_segment"](
                "unused", segments, 1, object(), 24.0, 0
            )


if __name__ == "__main__":
    unittest.main()
