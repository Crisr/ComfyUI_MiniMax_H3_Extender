"""Tests for the matched preview path of the Image Color Match.

The preview must show the same match as the final sidecars: the clip-by-clip
autosave that lands in the output folder is a copy of the assembled preview, so
a neutral preview means the user never sees the match in that file.

The H3 Extender modules cannot be imported outside ComfyUI, so these tests load
only the source functions they exercise out of ``motion_context_disk.py``.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_color_match import HELPER_NAMES, _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PREVIEW_NAMES = HELPER_NAMES | {
    "FULL_BATCH_FINAL_PROFILE_VERSION",
    "FULL_BATCH_H264_CACHE_CRF",
    "FULL_BATCH_H264_CACHE_PRESET",
    "normalize_full_batch_export_profile",
    "_matched_preview_clip_dir",
    "_matched_preview_clip_path",
    "_matched_source_preview_video_path",
    "_encode_matched_preview_clip_video",
    "_ensure_matched_source_preview_video",
    "_ensure_matched_preview_clip",
    "_encode_matched_source_video",
    "_raw_bytes_to_frames",
    "_iter_raw_frame_batches",
    "_matched_preview_profile",
}


class _FakeDecoder:
    def __init__(self, payload):
        self.stdout = io.BytesIO(payload)

    def wait(self):
        return 0

    def kill(self):
        return None


class _FakeSubprocess:
    """Minimal stand-in for the subprocess module used by the match encoders."""

    PIPE = -1
    DEVNULL = -3

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def Popen(self, command, **kwargs):
        self.calls.append(command)
        return _FakeDecoder(self.payload)

    def run(self, *args, **kwargs):
        return None


class MatchedPreviewFixtures(unittest.TestCase):
    """The preview fixtures the Image Color Match needs."""

    WIDTH = 6
    HEIGHT = 4
    # A non-clipping reference: the mean is only pinned to the reference when
    # the transfer output stays inside [0, 1].  With a dark reference the
    # clamp shifts the measured mean up, exactly like comfyui_essentials.
    REFERENCE_MEAN = [0.55, 0.45, 0.35]
    REFERENCE_STD = [0.12, 0.10, 0.08]

    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], PREVIEW_NAMES)
        self.ns["Path"] = Path
        self.ns["np"] = np
        self.ns["os"] = os
        self.ns["uuid"] = uuid
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_path = self.tmp / "chain.h3cache"
        self.data_path.write_bytes(b"cache")
        self.ns["_ensure_cache_root"] = lambda: self.tmp
        self.ns["_source_meta"] = lambda manifest: {
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "fps": 24.0,
        }
        self.ns["_source_working_path"] = lambda data_path: Path(
            data_path
        ).with_suffix(".source.mkv")
        self.written = []
        self.ns["_write_image_frames"] = self._record_frames
        self.started = []
        self.ns["_start_video_encoder"] = self._start_video_encoder
        self.ns["_finish_process"] = lambda *args, **kwargs: None
        self.blob_copies = []
        self.ns["_copy_blob_to_file"] = self._copy_blob_to_file
        self.fake_subprocess = None

    def _record_frames(self, proc, images, *args, **kwargs):
        self.written.append(images.detach().clone())

    def _start_video_encoder(
        self,
        ffmpeg,
        target,
        width,
        height,
        fps,
        codec,
        crf,
        preset,
        log_path,
        video_filter=None,
    ):
        self.started.append({"target": Path(target), "width": width, "height": height})
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"encoded")
        return None, None

    def _copy_blob_to_file(self, data_path, blob, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"neutral")
        self.blob_copies.append((blob, Path(target).name))

    # -- accessors -------------------------------------------------------
    def helper(self, name):
        func = self.ns.get(name)
        self.assertIsNotNone(func, f"{name} is missing from motion_context_disk.py")
        return func

    def match_spec(self, **overrides):
        spec = {
            "enabled": True,
            "space": "RGB",
            "factor": 1.0,
            "process_clip0": False,
            "reference_signature": "reference-a",
            "reference_mean": list(self.REFERENCE_MEAN),
            "reference_std": list(self.REFERENCE_STD),
        }
        spec.update(overrides)
        return self.helper("_normalize_color_match")(spec)

    def install_decoder(self, frame_count):
        frames = np.random.randint(
            0, 256, size=(frame_count, self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.fake_subprocess = _FakeSubprocess(frames.tobytes())
        self.ns["subprocess"] = self.fake_subprocess
        return frames

    def segments_with_blob(self, blob=b"blob", **extra):
        desc = {"decoded_mp4_blob": blob}
        desc.update(extra)
        return [desc]

    def manifest(self, **extra):
        manifest = {"base_resolution": {"width": self.WIDTH, "height": self.HEIGHT}}
        manifest.update(extra)
        return manifest

    def assert_matched_means(self, frames, places=3):
        means = frames.reshape(-1, 3).mean(dim=0)
        for channel, expected in enumerate(self.REFERENCE_MEAN):
            self.assertAlmostEqual(
                float(means[channel]),
                expected,
                places=places,
                msg=f"channel {channel} was not matched to the reference",
            )

    # -- tests -----------------------------------------------------------
    def test_inactive_match_reports_no_matched_paths(self):
        self.assertIsNone(
            self.helper("_matched_preview_clip_path")(self.data_path, 0, {})
        )
        self.assertIsNone(
            self.helper("_matched_source_preview_video_path")(self.data_path, {})
        )
        self.assertIsNone(
            self.helper("_ensure_matched_preview_clip")(
                self.data_path,
                self.segments_with_blob(),
                0,
                self.manifest(),
                "ffmpeg",
                24.0,
                "tok",
                {},
            )
        )
        self.assertIsNone(
            self.helper("_ensure_matched_source_preview_video")(
                self.data_path, self.manifest(), "ffmpeg", "tok", {}
            )
        )
        self.assertFalse((self.tmp / "chain.preview.match").exists())
        self.assertEqual(self.started, [])

    def test_matched_clip_path_is_keyed_by_the_match_signature(self):
        spec = self.match_spec()
        tag = self.helper("_color_match_signature")(spec)
        path = self.helper("_matched_preview_clip_path")(self.data_path, 0, spec)
        self.assertEqual(
            path.parent, self.helper("_matched_preview_clip_dir")(self.data_path)
        )
        self.assertEqual(path.parent, self.tmp / "chain.preview.match")
        self.assertEqual(path.name, f"preview_0000_{tag}.mp4")
        self.assertEqual(
            path, self.helper("_matched_preview_clip_path")(self.data_path, 0, spec)
        )
        other = self.helper("_matched_preview_clip_path")(
            self.data_path, 0, self.match_spec(factor=0.5)
        )
        self.assertNotEqual(path.name, other.name)
        third = self.helper("_matched_preview_clip_path")(self.data_path, 2, spec)
        self.assertEqual(third.name, f"preview_0002_{tag}.mp4")

    def test_source_preview_path_follows_the_clip0_switch(self):
        off = self.match_spec(process_clip0=False)
        self.assertIsNone(
            self.helper("_matched_source_preview_video_path")(self.data_path, off)
        )
        on = self.match_spec(process_clip0=True)
        tag = self.helper("_color_match_source_signature")(on)
        path = self.helper("_matched_source_preview_video_path")(self.data_path, on)
        self.assertTrue(path.name.endswith(f".source.preview.match.{tag}.mp4"))
        self.assertEqual(path.parent, self.tmp)
        other = self.helper("_matched_source_preview_video_path")(
            self.data_path, self.match_spec(process_clip0=True, factor=0.25)
        )
        self.assertNotEqual(path, other)

    def test_ensure_matched_clip_reuses_an_existing_file(self):
        spec = self.match_spec()
        target = self.helper("_matched_preview_clip_path")(self.data_path, 0, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"already matched")
        found = self.helper("_ensure_matched_preview_clip")(
            self.data_path,
            self.segments_with_blob(),
            0,
            self.manifest(),
            "ffmpeg",
            24.0,
            "tok",
            spec,
        )
        self.assertEqual(found, target)
        self.assertEqual(self.started, [])
        self.assertEqual(self.blob_copies, [])

    def test_ensure_matched_clip_builds_a_matched_preview(self):
        spec = self.match_spec()
        self.install_decoder(5)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            target = self.helper("_ensure_matched_preview_clip")(
                self.data_path,
                self.segments_with_blob(),
                0,
                self.manifest(),
                "ffmpeg",
                24.0,
                "tok",
                spec,
            )
        self.assertTrue(target.exists())
        self.assertEqual(self.blob_copies[0][1], "_tok_match_0000.mp4")
        self.assertEqual(len(self.started), 1)
        self.assertEqual(
            (self.started[0]["width"], self.started[0]["height"]),
            (self.WIDTH, self.HEIGHT),
        )
        self.assertEqual(len(self.written), 1)
        self.assert_matched_means(self.written[0])
        log = out.getvalue().lower()
        self.assertIn("image color match", log)
        self.assertIn("preview clip 1", log)

    def test_ensure_matched_clip_returns_none_without_a_neutral_blob(self):
        spec = self.match_spec()
        self.install_decoder(5)
        found = self.helper("_ensure_matched_preview_clip")(
            self.data_path,
            [{"decoded_mp4_blob": None}],
            0,
            self.manifest(),
            "ffmpeg",
            24.0,
            "tok",
            spec,
        )
        self.assertIsNone(found)
        self.assertEqual(self.started, [])
        target = self.helper("_matched_preview_clip_path")(self.data_path, 0, spec)
        self.assertFalse(target.exists())

    def test_chunked_matched_encoder_matches_every_chunk(self):
        spec = self.match_spec()
        frames = torch.rand(70, self.HEIGHT, self.WIDTH, 3)
        before = frames.clone()
        target = self.tmp / "matched.mp4"
        self.helper("_encode_matched_preview_clip_video")(
            "ffmpeg", frames, 24.0, target, "tok", spec
        )
        self.assertEqual([int(chunk.shape[0]) for chunk in self.written], [32, 32, 6])
        for chunk in self.written:
            self.assert_matched_means(chunk, places=3)
        self.assertTrue(torch.equal(frames, before))
        self.assertEqual(len(self.started), 1)
        self.assertTrue(target.exists())

    def test_source_preview_matched_file_is_reused_then_built(self):
        spec = self.match_spec(process_clip0=True)
        source = self.tmp / "chain.source.mkv"
        source.write_bytes(b"source")
        path_fn = self.helper("_matched_source_preview_video_path")
        ensure = self.helper("_ensure_matched_source_preview_video")
        target = path_fn(self.data_path, spec)
        target.write_bytes(b"already matched")
        self.assertEqual(
            ensure(self.data_path, self.manifest(), "ffmpeg", "tok", spec), target
        )
        self.assertEqual(self.started, [])
        target.unlink()
        self.install_decoder(5)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            built = ensure(self.data_path, self.manifest(), "ffmpeg", "tok", spec)
        self.assertEqual(built, target)
        self.assertTrue(built.exists())
        self.assertEqual(len(self.started), 1)
        self.assertEqual(len(self.written), 1)
        self.assert_matched_means(self.written[0])
        self.assertIn("clip 0 preview match", out.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
