"""Per-clip Motion Context: reference-echo head detection.

H3 ref2va packs the reference image immediately before the target's frame 0, so
a clip with no Motion Context opens on a held copy of that picture and then
hard-cuts into the real shot. These tests pin the detector that measures how
long that held prologue is, using synthetic frames shaped like real output.
"""

import ast
import unittest
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "motion_context_disk.py"
WANTED_NAMES = {
    "REF_ECHO_SCAN_FRAMES",
    "REF_ECHO_CUT_DELTA",
    "REF_ECHO_HOLD_DELTA",
}
WANTED_FUNCS = {"_detect_ref_echo_head"}


def _load_detector():
    """Extract the constants and the helper without importing ComfyUI."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id in WANTED_NAMES
                for target in node.targets
            ):
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
            body.append(node)
    if len(body) != len(WANTED_NAMES) + len(WANTED_FUNCS):
        raise AssertionError("motion_context_disk.py no longer defines the ref-echo detector.")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_detect_ref_echo_head"]


class RefEchoHeadTest(unittest.TestCase):
    def setUp(self):
        self.detect = _load_detector()

    @staticmethod
    def _clip(hold_frames, total=40):
        """A held reference still that cuts into a busy shot."""
        generator = torch.Generator().manual_seed(7)
        held = torch.rand(total, 32, 32, 3, generator=generator) * 0.2 + 0.4
        shot = torch.rand(total, 32, 32, 3, generator=generator) * 0.2 + 0.05
        frames = held[:hold_frames].clone()
        for step in range(hold_frames):
            frames[step] = held[0] + 0.002 * step
        return torch.cat((frames, shot[hold_frames:]), dim=0).clamp(0.0, 1.0)

    def test_measures_four_frame_prologue(self):
        self.assertEqual(self.detect(self._clip(4)), 4)

    def test_measures_seven_frame_prologue(self):
        self.assertEqual(self.detect(self._clip(7)), 7)

    def test_measures_single_frame_prologue(self):
        self.assertEqual(self.detect(self._clip(1)), 1)

    def test_normal_motion_is_not_trimmed(self):
        generator = torch.Generator().manual_seed(11)
        frames = torch.rand(40, 32, 32, 3, generator=generator)
        frames = frames * 0.05
        frames = frames + torch.arange(40).view(-1, 1, 1, 1) * 0.004
        self.assertEqual(self.detect(frames.clamp(0.0, 1.0)), 0)

    def test_slow_fade_is_not_trimmed(self):
        ramp = torch.arange(40).view(-1, 1, 1, 1).float() * 0.01
        self.assertEqual(self.detect(ramp.expand(40, 32, 32, 3).contiguous()), 0)

    def test_frozen_clip_without_cut_is_left_alone(self):
        still = torch.full((40, 32, 32, 3), 0.5)
        self.assertEqual(self.detect(still), 0)

    def test_prologue_longer_than_window_is_left_alone(self):
        self.assertEqual(self.detect(self._clip(30, total=40)), 0)


if __name__ == "__main__":
    unittest.main()
