"""CPU-only tests of Final Decode output naming; no ComfyUI or GPU required."""

import ast
import os
import re
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "motion_context_disk.py").read_text(encoding="utf-8")

FUNCTIONS = {
    "_safe_name",
    "_expand_filename_tokens",
    "_relative_output_parts",
    "_output_target",
    "_next_output_path",
    "_replace_locked_source",
}
CONSTANTS = {"_DATE_TOKEN", "_JAVA_DATE_PARTS"}


def _load_namespace():
    tree = ast.parse(SOURCE)
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") in CONSTANTS for target in node.targets)
        )
    ]

    class SilentLog:
        @staticmethod
        def warning(*args, **kwargs):
            return None

    namespace = {
        "re": re,
        "Path": Path,
        "datetime": datetime,
        "os": os,
        "time": __import__("time"),
        "uuid": uuid,
        "_LOG": SilentLog(),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "motion_context_disk.py", "exec"), namespace)
    return namespace


ns = _load_namespace()


class Tokens(unittest.TestCase):
    def test_date_token_expands_to_today(self):
        result = ns["_expand_filename_tokens"]("%date:yyyy-MM-dd%x3/H3_Clip")
        self.assertEqual(result, f"{datetime.now():%Y-%m-%d}x3/H3_Clip")

    def test_time_token_expands(self):
        result = ns["_expand_filename_tokens"]("%time:HH-mm-ss%_clip")
        self.assertRegex(result, r"^\d{2}-\d{2}-\d{2}_clip$")

    def test_unknown_tokens_stay_literal(self):
        self.assertEqual(ns["_expand_filename_tokens"]("%foo%bar"), "%foo%bar")

    def test_unclosed_token_stays_literal(self):
        self.assertEqual(ns["_expand_filename_tokens"]("%date:yyyy-MM-dd"), "%date:yyyy-MM-dd")

    def test_empty_pattern_stays_literal(self):
        self.assertEqual(ns["_expand_filename_tokens"]("%date:%"), "%date:%")


class Paths(unittest.TestCase):
    def test_subfolders_are_kept(self):
        parts = ns["_relative_output_parts"]("%date:yyyy-MM-dd%x3/H3_Clip")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1], "H3_Clip")

    def test_parent_traversal_is_dropped(self):
        parts = ns["_relative_output_parts"]("../../evil/clip")
        self.assertEqual(parts, ["evil", "clip"])

    def test_flat_prefix_keeps_the_old_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = ns["_output_target"](tmp, "H3_Clip", "mp4")
            self.assertEqual(Path(target), Path(tmp) / "H3_Clip.mp4")

    def test_dated_subfolder_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(ns["_output_target"](tmp, "%date:yyyy-MM-dd%/H3_Clip", "mp4"))
            self.assertEqual(target.parent.name, f"{datetime.now():%Y-%m-%d}")
            self.assertTrue(target.parent.is_dir())

    def test_numbering_skips_used_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(ns["_output_target"](tmp, "clip", "mp4"))
            first.write_bytes(b"x")
            second = ns["_next_output_path"](tmp, "clip", "mp4")
            self.assertEqual(Path(second).name, "clip_00001.mp4")


class LockedDestination(unittest.TestCase):
    class FakeOS:
        def __init__(self):
            self.attempts = []

        def replace(self, src, dst):
            name = Path(dst).name
            self.attempts.append(name)
            if name == "clip.mp4":
                raise PermissionError(5, "Access is denied")
            Path(dst).write_bytes(Path(src).read_bytes())
            Path(src).unlink()

    def test_a_locked_file_falls_back_to_the_next_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / ".clip.tmp.mp4"
            tmp_file.write_bytes(b"payload")
            destination = Path(tmp) / "clip.mp4"
            destination.write_bytes(b"yesterday")
            fake = self.FakeOS()
            original = ns["os"]
            ns["os"] = fake
            try:
                result = Path(
                    ns["_replace_locked_source"](tmp_file, destination, attempts=2, delay=0)
                )
            finally:
                ns["os"] = original
            self.assertEqual(result.name, "clip_00001.mp4")
            self.assertEqual(result.read_bytes(), b"payload")
            self.assertFalse(tmp_file.exists())
            self.assertEqual(fake.attempts, ["clip.mp4", "clip.mp4", "clip_00001.mp4"])

    def test_a_free_file_keeps_its_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / ".clip.tmp.mp4"
            tmp_file.write_bytes(b"payload")
            destination = Path(tmp) / "clip.mp4"
            result = ns["_replace_locked_source"](tmp_file, destination, attempts=2, delay=0)
            self.assertEqual(Path(result), destination)
            self.assertEqual(destination.read_bytes(), b"payload")


class Wiring(unittest.TestCase):
    def test_the_autosave_uses_the_locked_destination_fallback(self):
        text = (ROOT / "motion_context_disk.py").read_text(encoding="utf-8")
        self.assertIn("return _replace_locked_source(tmp, destination)", text)
        self.assertIn("destination = _output_target(output_dir, filename_prefix, \"mp4\")", text)


if __name__ == "__main__":
    unittest.main()
