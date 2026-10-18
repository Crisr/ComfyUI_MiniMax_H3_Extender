"""CPU-only tests for the Image Color Match helpers in motion_context_disk.py.

The H3 Extender modules cannot be imported outside ComfyUI, so the tests load
only the source functions they exercise (same technique as
test_project_autosave.py) and run them with real torch/kornia.
"""
import ast
import contextlib
import io
import json
import hashlib
import math
import tempfile
import time
from pathlib import Path
import unittest

import kornia
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

HELPER_NAMES = {
    "COLOR_MATCH_SPACES",
    "COLOR_MATCH_STD_FLOOR",
    "COLOR_MATCH_CHUNK_FRAMES",
    "_COLOR_MATCH_TO_SPACE",
    "_COLOR_MATCH_FROM_SPACE",
    "_normalize_color_match",
    "_color_match_is_active",
    "_color_match_signature",
    "_color_match_source_signature",
    "_color_match_device",
    "_color_match_console",
    "_COLOR_MATCH_DEVICE",
    "_color_space_convert",
    "_color_match_reference_stats",
    "_apply_color_match_frames",
}

CACHE_KEY_NAMES = HELPER_NAMES | {
    "FULL_BATCH_FINAL_CACHE_VERSION",
    "FULL_BATCH_FINAL_PROFILE_VERSION",
    "normalize_full_batch_export_profile",
    "_full_batch_export_profile_signature",
    "_normalize_color_adjustment",
    "_color_adjustment_signature",
    "_final_segment_cache_meta_matches",
    "_tag_ref2va_final_segment_cache",
}

ENCODE_NAMES = HELPER_NAMES | {
    "FULL_BATCH_FINAL_PROFILE_VERSION",
    "normalize_full_batch_export_profile",
    "_normalize_color_adjustment",
    "_color_is_neutral",
    "_ffmpeg_color_filter",
    "_encode_final_segment_video",
}

SOURCE_NAMES = HELPER_NAMES | {
    "FULL_BATCH_FINAL_PROFILE_VERSION",
    "normalize_full_batch_export_profile",
    "_full_batch_export_profile_signature",
    "_full_batch_export_profile_extension",
    "_final_segment_cache_dir",
    "_source_final_video_path",
    "_raw_bytes_to_frames",
    "_iter_raw_frame_batches",
}


def _load(filenames, names):
    """Exec the requested module-level assignments/functions into one namespace."""
    ns = {
        "json": json,
        "hashlib": hashlib,
        "math": math,
        "torch": torch,
        "kornia": kornia,
    }
    for filename in filenames:
        tree = ast.parse((ROOT / filename).read_text())
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets
            ):
                body.append(node)
        if body:
            exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), ns)
    return ns


def _essentials_color_match(image, reference, color_space, factor):
    """Literally the comfyui_essentials ImageColorMatch.execute() math."""
    image = image.permute([0, 3, 1, 2])
    reference = reference.permute([0, 3, 1, 2])

    to_space = {
        "LAB": kornia.color.rgb_to_lab,
        "YCbCr": kornia.color.rgb_to_ycbcr,
        "LUV": kornia.color.rgb_to_luv,
        "YUV": kornia.color.rgb_to_yuv,
        "XYZ": kornia.color.rgb_to_xyz,
    }
    from_space = {
        "LAB": kornia.color.lab_to_rgb,
        "YCbCr": kornia.color.ycbcr_to_rgb,
        "LUV": kornia.color.luv_to_rgb,
        "YUV": kornia.color.yuv_to_rgb,
        "XYZ": kornia.color.xyz_to_rgb,
    }
    if color_space in to_space:
        reference = to_space[color_space](reference)
    reference_mean = reference.mean(dim=[2, 3], keepdim=True)
    reference_std = reference.std(dim=[2, 3], keepdim=True)

    if color_space in to_space:
        image = to_space[color_space](image)
    image_mean = image.mean(dim=[2, 3], keepdim=True)
    image_std = image.std(dim=[2, 3], keepdim=True)

    matched = torch.nan_to_num((image - image_mean) / image_std) * torch.nan_to_num(reference_std) + reference_mean
    matched = factor * matched + (1 - factor) * image
    if color_space in from_space:
        matched = from_space[color_space](matched)
    return matched.permute([0, 2, 3, 1]).clamp(0, 1)


class ColorMatchHelpers(unittest.TestCase):
    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], HELPER_NAMES)

    def normalize(self, value=None):
        return self.ns["_normalize_color_match"](value)

    def signature(self, value=None):
        return self.ns["_color_match_signature"](value)

    def source_signature(self, value=None):
        return self.ns["_color_match_source_signature"](value)

    def apply(self, frames, value):
        return self.ns["_apply_color_match_frames"](frames, value, device=torch.device("cpu"))

    def reference_stats(self, reference, space="LAB"):
        return self.ns["_color_match_reference_stats"](
            reference, space, device=torch.device("cpu")
        )

    # -- normalization and signature -------------------------------------

    def test_defaults_match_the_documented_widget_defaults(self):
        spec = self.normalize()
        self.assertEqual(spec["space"], "LAB")
        self.assertEqual(spec["factor"], 1.0)
        self.assertIs(spec["process_clip0"], False)
        self.assertIs(spec["enabled"], False)
        self.assertEqual(spec["reference_mean"], [])
        self.assertEqual(spec["reference_std"], [])

    def test_unknown_space_and_out_of_range_factor_are_clamped(self):
        spec = self.normalize({"space": "CMYK", "factor": 7.5, "enabled": True})
        self.assertEqual(spec["space"], "LAB")
        self.assertEqual(spec["factor"], 1.0)
        self.assertEqual(self.normalize({"factor": -3})["factor"], 0.0)

    def test_signature_changes_with_every_documented_setting(self):
        base = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "process_clip0": False,
            "reference_signature": "abc",
            "reference_mean": [0.1, 0.2, 0.3],
            "reference_std": [0.4, 0.5, 0.6],
        }
        reference = self.signature(base)
        self.assertTrue(reference)
        self.assertEqual(reference, self.signature(dict(base)))
        for key, value in [
            ("space", "RGB"),
            ("factor", 0.5),
            ("reference_signature", "def"),
            ("reference_mean", [0.2, 0.2, 0.3]),
            ("enabled", False),
        ]:
            changed = dict(base)
            changed[key] = value
            self.assertNotEqual(reference, self.signature(changed), key)

    def test_clip0_toggle_never_invalidates_generated_clips(self):
        base = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "reference_mean": [0.1, 0.2, 0.3],
            "reference_std": [0.4, 0.5, 0.6],
        }
        self.assertEqual(self.signature(base), self.signature({**base, "process_clip0": True}))
        self.assertNotEqual(
            self.source_signature(base), self.source_signature({**base, "process_clip0": True})
        )
        self.assertEqual(self.source_signature({}), "")
        self.assertEqual(self.source_signature({"enabled": True}), "")

    def test_signature_is_empty_when_the_match_is_unavailable(self):
        self.assertEqual(self.signature(), "")
        self.assertEqual(self.signature({"enabled": False, "reference_mean": [0.1, 0.2, 0.3], "reference_std": [0.2, 0.3, 0.4]}), "")
        self.assertEqual(self.signature({"enabled": True}), "")

    # -- reference statistics --------------------------------------------

    def test_reference_stats_give_one_profile_per_channel(self):
        torch.manual_seed(5)
        reference = torch.rand(1, 4, 6, 3)
        mean, std = self.reference_stats(reference)
        self.assertEqual(mean.shape, (1, 3, 1, 1))
        self.assertEqual(std.shape, (1, 3, 1, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())

    def test_reference_stats_average_a_multi_image_reference(self):
        torch.manual_seed(6)
        first = torch.rand(1, 5, 5, 3) * 0.5 + 0.25
        second = torch.rand(1, 5, 5, 3) * 0.5 + 0.25
        repeated_mean, repeated_std = self.reference_stats(torch.cat([first, first], dim=0))
        single_mean, single_std = self.reference_stats(first)
        self.assertTrue(torch.allclose(repeated_mean, single_mean, atol=1e-5))
        self.assertTrue(torch.allclose(repeated_std, single_std, atol=1e-5))

        first_mean, _ = self.reference_stats(first)
        second_mean, _ = self.reference_stats(second)
        multi_mean, _ = self.reference_stats(torch.cat([first, second], dim=0))
        low = torch.minimum(first_mean, second_mean)
        high = torch.maximum(first_mean, second_mean)
        self.assertTrue(bool(((multi_mean >= low - 1e-5) & (multi_mean <= high + 1e-5)).all()))

    # -- the transform itself --------------------------------------------

    def test_match_reproduces_the_essentials_formula(self):
        torch.manual_seed(7)
        reference = torch.rand(1, 12, 10, 3) * 0.6 + 0.2
        frames = torch.rand(4, 8, 8, 3) * 0.4 + 0.1
        for space, tolerance in [("RGB", 1e-6), ("LAB", 1e-4), ("YCbCr", 1e-5), ("XYZ", 1e-4)]:
            with self.subTest(space=space):
                spec = {
                    "enabled": True,
                    "space": space,
                    "factor": 1.0,
                    "process_clip0": False,
                    "reference_signature": "ref",
                    "reference_mean": [0.0],
                    "reference_std": [0.0],
                }
                mean, std = self.reference_stats(reference, space)
                spec["reference_mean"] = [float(v) for v in mean.reshape(-1)]
                spec["reference_std"] = [float(v) for v in std.reshape(-1)]
                wanted = _essentials_color_match(frames, reference, space, 1.0)
                got = self.apply(frames, spec)
                self.assertEqual(got.shape, frames.shape)
                self.assertTrue(torch.allclose(got, wanted, atol=tolerance), space)

    def test_match_is_chunk_size_independent(self):
        torch.manual_seed(11)
        reference = torch.rand(1, 6, 5, 3)
        frames = torch.rand(70, 4, 4, 3)
        mean, std = self.reference_stats(reference, "LAB")
        spec = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "reference_mean": [float(v) for v in mean.reshape(-1)],
            "reference_std": [float(v) for v in std.reshape(-1)],
        }
        chunked = self.apply(frames, spec)
        whole = _essentials_color_match(frames, reference, "LAB", 1.0)
        self.assertTrue(torch.allclose(chunked, whole, atol=1e-5))

    def test_factor_zero_returns_the_input_colour(self):
        torch.manual_seed(3)
        reference = torch.rand(1, 5, 5, 3)
        frames = torch.rand(3, 6, 6, 3)
        for space, tolerance in [("RGB", 1e-6), ("LAB", 1e-3)]:
            with self.subTest(space=space):
                mean, std = self.reference_stats(reference, space)
                spec = {
                    "enabled": True,
                    "space": space,
                    "factor": 0.0,
                    "reference_mean": [float(v) for v in mean.reshape(-1)],
                    "reference_std": [float(v) for v in std.reshape(-1)],
                }
                self.assertTrue(torch.allclose(self.apply(frames, spec), frames, atol=tolerance))

    def test_disabled_or_incomplete_match_returns_the_input_untouched(self):
        frames = torch.rand(2, 4, 4, 3)
        for spec in [None, {}, {"enabled": False, "reference_mean": [0.1] * 3, "reference_std": [0.2] * 3},
                     {"enabled": True, "reference_mean": [0.1] * 3, "reference_std": []}]:
            with self.subTest(spec=spec):
                out = self.apply(frames, spec)
                self.assertTrue(torch.equal(out, frames))

    def test_flat_frame_does_not_blow_up(self):
        reference = torch.rand(1, 4, 4, 3) * 0.5 + 0.25
        frames = torch.full((2, 4, 4, 3), 0.4)
        mean, std = self.reference_stats(reference, "LAB")
        spec = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "reference_mean": [float(v) for v in mean.reshape(-1)],
            "reference_std": [float(v) for v in std.reshape(-1)],
        }
        out = self.apply(frames, spec)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_single_channel_input_is_rejected_with_a_clear_error(self):
        frames = torch.rand(2, 4, 4, 1)
        with self.assertRaises(ValueError):
            self.apply(frames, {"enabled": True, "reference_mean": [0.1] * 3, "reference_std": [0.2] * 3})


class FinalSegmentCacheKeys(unittest.TestCase):
    """The match spec must participate in the final sidecar validity key."""

    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], CACHE_KEY_NAMES)
        self.ns["Path"] = Path
        self.ns["time"] = time
        self.ns["_write_json_atomic"] = lambda path, payload: None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ref2va_0000.mp4"
        self.path.write_bytes(b"not really a video")
        self.profile = self.ns["normalize_full_batch_export_profile"]({"codec": "H.264"})
        self.spec = self._spec()

    def _spec(self, **overrides):
        spec = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "process_clip0": False,
            "reference_signature": "ref-abc",
            "reference_mean": [0.1, 0.2, 0.3],
            "reference_std": [0.05, 0.06, 0.07],
        }
        spec.update(overrides)
        return spec

    def _desc(self, **extra):
        desc = {
            "final_video_cache_version": int(self.ns["FULL_BATCH_FINAL_CACHE_VERSION"]),
            "final_video_profile_signature": self.ns["_full_batch_export_profile_signature"](
                self.profile
            ),
            "final_video_color_signature": self.ns["_color_adjustment_signature"](None),
        }
        desc.update(extra)
        return desc

    def matches(self, desc, color_match=None):
        return self.ns["_final_segment_cache_meta_matches"](
            desc, self.profile, None, self.path, color_match=color_match
        )

    def tag(self, color_match=None):
        manifest = {"segments": [self._desc()]}
        _, desc = self.ns["_tag_ref2va_final_segment_cache"](
            Path(self.tmp.name) / "manifest.json",
            manifest,
            0,
            self.profile,
            None,
            color_match=color_match,
        )
        return desc

    def test_sidecar_tagged_before_the_match_existed_still_matches_while_off(self):
        self.assertTrue(self.matches(self._desc()))
        self.assertTrue(self.matches(self._desc(), color_match=self._spec(enabled=False)))

    def test_enabling_the_match_invalidates_a_sidecar_tagged_before_it(self):
        self.assertFalse(self.matches(self._desc(), color_match=self.spec))

    def test_tagged_sidecar_matches_only_its_own_match_spec(self):
        desc = self.tag(self.spec)
        self.assertTrue(self.matches(desc, self.spec))
        self.assertFalse(self.matches(desc, self._spec(factor=0.5)))
        self.assertFalse(self.matches(desc, self._spec(space="YCbCr")))
        self.assertFalse(self.matches(desc, self._spec(reference_signature="ref-other")))
        self.assertFalse(self.matches(desc, self._spec(reference_mean=[0.4, 0.5, 0.6])))
        self.assertFalse(self.matches(desc))

    def test_clip0_toggle_never_invalidates_generated_clips(self):
        desc = self.tag(self._spec(process_clip0=True))
        self.assertTrue(self.matches(desc, self._spec(process_clip0=False)))
        self.assertTrue(self.matches(desc, self._spec(process_clip0=True)))

    def test_tag_records_the_normalized_match_spec(self):
        desc = self.tag({"enabled": True, "space": "nonsense", "factor": 7.0})
        self.assertEqual(desc["final_video_match_signature"], "")
        self.assertEqual(desc["color_match"]["space"], "LAB")
        self.assertEqual(desc["color_match"]["factor"], 1.0)
        tagged = self.tag(self.spec)
        self.assertEqual(
            tagged["final_video_match_signature"],
            self.ns["_color_match_signature"](self.spec),
        )
        self.assertNotEqual(tagged["final_video_match_signature"], "")


class FinalEncodeAppliesTheMatch(unittest.TestCase):
    """The one lossy final encode must see the matched frames."""

    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], ENCODE_NAMES)
        self.ns["Path"] = Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.written = []

        def _write_image_frames(proc, images, batch_frames=8):
            self.written.append(images)

        self.ns["_ensure_cache_root"] = lambda: Path(self.tmp.name)
        self.ns["_start_video_encoder"] = lambda *args, **kwargs: (None, None)
        self.ns["_finish_process"] = lambda *args, **kwargs: None
        self.ns["_write_image_frames"] = _write_image_frames

    def _spec(self, **overrides):
        reference = torch.rand(1, 16, 16, 3) * 0.6 + 0.2
        mean, std = self.ns["_color_match_reference_stats"](
            reference, "RGB", device=torch.device("cpu")
        )
        spec = {
            "enabled": True,
            "space": "RGB",
            "factor": 1.0,
            "reference_signature": "ref",
            "reference_mean": [float(v) for v in mean.reshape(-1)],
            "reference_std": [float(v) for v in std.reshape(-1)],
        }
        spec.update(overrides)
        return spec

    def encode(self, frames, color_match=None):
        profile = self.ns["normalize_full_batch_export_profile"]({"codec": "H.264"})
        self.ns["_encode_final_segment_video"](
            "ffmpeg",
            frames,
            24.0,
            Path(self.tmp.name) / "out.mp4",
            "token",
            profile,
            None,
            color_match=color_match,
        )
        return self.written[-1]

    def test_the_encoder_receives_the_matched_frames(self):
        frames = torch.rand(3, 8, 8, 3)
        spec = self._spec()
        original = frames.clone()
        seen = self.encode(frames, spec)
        expected = self.ns["_apply_color_match_frames"](
            original, spec, device=torch.device("cpu")
        )
        self.assertTrue(torch.allclose(seen, expected, atol=1e-6))
        self.assertFalse(torch.allclose(seen, original, atol=1e-3))

    def test_the_input_tensor_is_left_untouched(self):
        frames = torch.rand(3, 8, 8, 3)
        original = frames.clone()
        self.encode(frames, self._spec())
        self.assertTrue(torch.equal(frames, original))

    def test_an_inactive_match_streams_the_same_tensor(self):
        frames = torch.rand(3, 8, 8, 3)
        seen = self.encode(frames)
        self.assertIs(seen, frames)

    def test_the_match_reports_start_and_stop_on_the_console(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.encode(torch.rand(3, 8, 8, 3), self._spec())
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        starts = [line for line in lines if "start" in line]
        stops = [line for line in lines if "stop" in line]
        self.assertEqual(len(starts), 1, lines)
        self.assertEqual(len(stops), 1, lines)
        self.assertIn("image color match", starts[0].lower())
        self.assertIn("out.mp4", starts[0])
        self.assertIn("image color match", stops[0].lower())

    def test_an_inactive_match_stays_silent(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.encode(torch.rand(3, 8, 8, 3))
        self.assertEqual(buffer.getvalue().strip(), "")


class Clip0SourceMatchLogging(unittest.TestCase):
    """The Clip 0 re-encode must report its start and its stop."""

    def setUp(self):
        names = SOURCE_NAMES | {"_encode_matched_source_video"}
        self.ns = _load(["motion_context_disk.py"], names)
        self.ns["np"] = np
        self.ns["Path"] = Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "clip.source.mkv"
        self.source.write_bytes(b"clip0")
        self.target = Path(self.tmp.name) / "source_matched.mp4"
        self.profile = self.ns["normalize_full_batch_export_profile"]({"codec": "H.264"})
        self.frames = np.random.randint(0, 256, size=(5, 3, 4, 3), dtype=np.uint8)
        self.written = []

        class _FakeDecoder:
            def __init__(self, stream):
                self.stdout = stream

            def wait(self):
                return 0

            def kill(self):
                return None

        class _FakeSubprocess:
            PIPE = -1
            DEVNULL = -3

            def __init__(self, stream):
                self.stream = stream

            def Popen(self, args, **kwargs):
                return _FakeDecoder(io.BytesIO(self.stream))

            def run(self, *args, **kwargs):
                return None

        self.ns["subprocess"] = _FakeSubprocess(self.frames.tobytes())
        self.ns["_ensure_cache_root"] = lambda: Path(self.tmp.name)
        self.ns["_start_video_encoder"] = lambda *args, **kwargs: (None, None)
        self.ns["_finish_process"] = lambda *args, **kwargs: None
        self.ns["_write_image_frames"] = (
            lambda proc, images, batch_frames=8: self.written.append(images)
        )

    def _spec(self, **overrides):
        spec = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "process_clip0": True,
            "reference_signature": "ref",
            "reference_mean": [0.1, 0.2, 0.3],
            "reference_std": [0.05, 0.06, 0.07],
        }
        spec.update(overrides)
        return spec

    def match(self, color_match):
        return self.ns["_encode_matched_source_video"](
            "ffmpeg",
            self.source,
            self.target,
            4,
            3,
            24.0,
            self.profile,
            "token",
            color_match,
        )

    def test_the_source_match_reports_start_and_stop(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.match(self._spec())
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        starts = [line for line in lines if "start" in line]
        stops = [line for line in lines if "stop" in line]
        self.assertEqual(len(starts), 1, lines)
        self.assertEqual(len(stops), 1, lines)
        self.assertIn("image color match", starts[0].lower())
        self.assertIn("clip 0", starts[0].lower())
        self.assertIn("5", stops[0])

    def test_no_match_means_no_logging(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.match(None)
        self.assertEqual(buffer.getvalue().strip(), "")


class Clip0FinalNaming(unittest.TestCase):
    """Clip 0 gets its own final file only when the match includes it."""

    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], SOURCE_NAMES)
        self.ns["np"] = np
        self.ns["Path"] = Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "clip.h3cache"
        self.profile = self.ns["normalize_full_batch_export_profile"]({"codec": "H.264"})
        self.legacy = self.ns["_source_final_video_path"](self.data_path, self.profile)

    def _spec(self, **overrides):
        spec = {
            "enabled": True,
            "space": "LAB",
            "factor": 1.0,
            "process_clip0": True,
            "reference_signature": "ref",
            "reference_mean": [0.1, 0.2, 0.3],
            "reference_std": [0.05, 0.06, 0.07],
        }
        spec.update(overrides)
        return spec

    def path(self, spec=None):
        return self.ns["_source_final_video_path"](self.data_path, self.profile, spec)

    def test_the_legacy_name_survives_a_disabled_or_excluded_match(self):
        self.assertEqual(self.path(None), self.legacy)
        self.assertEqual(self.path({}), self.legacy)
        self.assertEqual(self.path(self._spec(enabled=False)), self.legacy)
        self.assertEqual(self.path(self._spec(process_clip0=False)), self.legacy)

    def test_an_included_match_gets_a_distinct_name(self):
        matched = self.path(self._spec())
        self.assertNotEqual(matched, self.legacy)
        self.assertEqual(matched.parent, self.legacy.parent)
        self.assertTrue(matched.name.startswith("source_"))
        self.assertTrue(matched.name.endswith(".mp4"))
        self.assertNotEqual(matched, self.path(self._spec(factor=0.5)))
        self.assertNotEqual(matched, self.path(self._spec(space="YCbCr")))
        self.assertEqual(matched, self.path(self._spec(process_clip0=True)))


class RawFrameBatching(unittest.TestCase):
    """Clip 0 decode must stream in bounded batches and honour frame bounds."""

    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], SOURCE_NAMES)
        self.ns["np"] = np

    def test_batches_split_on_frame_boundaries_and_keep_the_order(self):
        raw = np.random.randint(0, 256, size=(70, 3, 4, 3), dtype=np.uint8)
        batches = list(
            self.ns["_iter_raw_frame_batches"](io.BytesIO(raw.tobytes()), 4, 3, 32)
        )
        self.assertEqual([int(b.shape[0]) for b in batches], [32, 32, 6])
        seen = 0
        for batch in batches:
            count = int(batch.shape[0])
            expected = torch.from_numpy(raw[seen:seen + count].copy()).float().div_(255.0)
            self.assertTrue(torch.equal(batch, expected))
            self.assertEqual(tuple(batch.shape[1:]), (3, 4, 3))
            seen += count

    def test_a_torn_trailing_frame_is_dropped(self):
        raw = np.random.randint(0, 256, size=(5, 2, 2, 3), dtype=np.uint8)
        stream = io.BytesIO(raw.tobytes() + b"\x01\x02")
        batches = list(self.ns["_iter_raw_frame_batches"](stream, 2, 2, 4))
        self.assertEqual([int(b.shape[0]) for b in batches], [4, 1])

    def test_an_empty_stream_yields_nothing(self):
        self.assertEqual(list(self.ns["_iter_raw_frame_batches"](io.BytesIO(b""), 2, 2, 4)), [])


EXTENDER_NAMES = {
    "_color_match_reference_signature",
    "_build_color_match",
}


class ColorMatchSpecFromImage(unittest.TestCase):
    """extender._build_color_match turns an IMAGE socket into the match spec."""

    def setUp(self):
        self.ns = _load(["extender.py"], EXTENDER_NAMES)
        self.ns.update(_load(["motion_context_disk.py"], HELPER_NAMES))
        self.ns["hashlib"] = hashlib
        self.ns["torch"] = torch
        self.image = torch.rand(1, 8, 8, 3)

    def build(self, image, **kwargs):
        return self.ns["_build_color_match"](image, **kwargs)

    def test_no_image_is_inert(self):
        for value in (None, {}, [], ""):
            spec = self.build(value)
            self.assertFalse(spec["enabled"])
            self.assertFalse(spec["process_clip0"])
            self.assertEqual(spec["reference_mean"], [])
            self.assertEqual(spec["reference_std"], [])
            self.assertEqual(self.ns["_color_match_signature"](spec), "")

    def test_a_connected_image_enables_the_match(self):
        spec = self.build(self.image)
        self.assertTrue(spec["enabled"])
        self.assertTrue(self.ns["_color_match_is_active"](spec))
        self.assertEqual(spec["space"], "LAB")
        self.assertEqual(spec["factor"], 1.0)
        self.assertFalse(spec["process_clip0"])
        self.assertEqual(len(spec["reference_mean"]), 3)
        self.assertEqual(len(spec["reference_std"]), 3)
        self.assertTrue(all(math.isfinite(x) for x in spec["reference_mean"]))
        self.assertTrue(spec["reference_signature"])
        self.assertTrue(self.ns["_color_match_signature"](spec))

    def test_options_reach_the_spec(self):
        spec = self.build(self.image, space="YCbCr", factor=0.5, process_clip0=True)
        self.assertEqual(spec["space"], "YCbCr")
        self.assertEqual(spec["factor"], 0.5)
        self.assertTrue(spec["process_clip0"])

    def test_unknown_options_fall_back(self):
        spec = self.build(self.image, space="nope", factor=7.0)
        self.assertEqual(spec["space"], "LAB")
        self.assertEqual(spec["factor"], 1.0)

    def test_a_different_image_changes_the_signature(self):
        first = self.build(torch.zeros(1, 8, 8, 3))
        second = self.build(torch.ones(1, 8, 8, 3))
        self.assertNotEqual(first["reference_signature"], second["reference_signature"])
        self.assertNotEqual(
            self.ns["_color_match_signature"](first),
            self.ns["_color_match_signature"](second),
        )

    def test_a_repeated_image_keeps_the_signature(self):
        first = self.build(torch.full((1, 8, 8, 3), 0.25))
        second = self.build(torch.full((1, 8, 8, 3), 0.25) + 0.0)
        self.assertEqual(first["reference_signature"], second["reference_signature"])
        self.assertEqual(
            self.ns["_color_match_signature"](first),
            self.ns["_color_match_signature"](second),
        )

    def test_the_clip0_toggle_never_changes_the_clip_signature(self):
        off = self.build(self.image, process_clip0=False)
        on = self.build(self.image, process_clip0=True)
        self.assertEqual(
            self.ns["_color_match_signature"](off),
            self.ns["_color_match_signature"](on),
        )

    def test_a_malformed_image_is_inert(self):
        for value in (torch.rand(8, 8, 3), torch.rand(1, 8, 8, 1), torch.rand(1, 8, 8, 4)):
            spec = self.build(value)
            self.assertFalse(spec["enabled"])
            self.assertEqual(self.ns["_color_match_signature"](spec), "")


if __name__ == "__main__":
    unittest.main()
