"""CPU-only tests of the per-clip Motion Context flag; no ComfyUI/GPU required."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    tree = ast.parse((ROOT / "extender.py").read_text())
    names = {"_parse_clips_json", "_segment_motion_mismatch"}
    ns = {
        "json": json,
        "MAX_CLIPS": 12,
        "DEFAULT_DURATION": 10.0,
        "DEFAULT_SEED_MAX": 2**53 - 1,
        "_default_clip": lambda index=0: {"id": f"clip_{index + 1}"},
        "_normalize_color_adjustment": lambda value=None: {},
        "_normalize_clip_loras": lambda value, legacy=None: [],
        "_normalize_local_refs": lambda value=None: {},
        "_normalize_ref_descriptor": lambda value: None,
        "_normalize_fl2va_guides": lambda value, legacy_frame=None, legacy_idx=0: [],
        "_normalize_generation_mode": lambda value: "fl2va" if str(value) == "fl2va" else "ref2va",
    }
    exec(
        compile(
            ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]),
            "extender.py",
            "exec",
        ),
        ns,
    )
    return ns["_parse_clips_json"], ns["_segment_motion_mismatch"]


parse, mismatch = _load()


class ClipMotionContext(unittest.TestCase):
    def test_first_clip_never_receives_motion_context(self):
        clips = parse(json.dumps([{"motion_context": True}, {"motion_context": True}]))
        self.assertEqual([c["motion_context"] for c in clips], [False, True])

    def test_missing_key_stays_on_for_legacy_payloads(self):
        clips = parse(json.dumps([{}, {}, {}]), generation_mode="ref2va", motion_context=True)
        self.assertEqual([c["motion_context"] for c in clips], [False, True, True])

    def test_off_state_survives_round_trip(self):
        raw = json.dumps([{"prompt": "a"}, {"prompt": "b", "motion_context": False}, {"prompt": "c"}])
        clips = parse(raw)
        self.assertEqual([c["motion_context"] for c in clips], [False, False, True])
        # The workflow widget stores {"version": 2, "clips": [...]}.
        again = parse(json.dumps({"version": 2, "clips": clips}))
        self.assertEqual([c["motion_context"] for c in again], [False, False, True])

    def test_non_boolean_values_follow_python_truthiness(self):
        clips = parse(json.dumps([{}, {"motion_context": "false"}, {"motion_context": 0}]))
        self.assertEqual([c["motion_context"] for c in clips], [False, True, False])

    def test_mismatch_reads_legacy_segments_as_motion_on(self):
        self.assertFalse(mismatch({"motion_context": True}, 0, False))
        self.assertFalse(mismatch({}, 2, True))
        self.assertTrue(mismatch({}, 2, False))
        self.assertTrue(mismatch({"motion_context": True}, 2, False))
        self.assertTrue(mismatch({"motion_context": False}, 2, True))
        self.assertFalse(mismatch({"motion_context": False}, 2, False))
        self.assertFalse(mismatch(None, 2, False))


if __name__ == "__main__":
    unittest.main()
