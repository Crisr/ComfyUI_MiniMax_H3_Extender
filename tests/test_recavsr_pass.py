"""CPU-only tests of the ReCaVSR export pass; no ComfyUI, GPU or weights needed."""

import ast
import importlib.util
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_pass():
    spec = importlib.util.spec_from_file_location("h3_recavsr_pass", ROOT / "recavsr_pass.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recavsr_pass = _load_pass()


class FakeVideo:
    def __init__(self, path):
        self._path = str(path)

    def get_stream_source(self):
        return self._path


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


class Settings(unittest.TestCase):
    def test_defaults(self):
        payload = recavsr_pass.normalize_settings(None)
        self.assertEqual(payload, {"enabled": False, "scale": 2.0, "color_fix": "none", "seed": 42})

    def test_clamps_scale_seed_and_colour(self):
        payload = recavsr_pass.normalize_settings(
            {"enabled": "true", "scale": 99, "color_fix": "sepia", "seed": -5}
        )
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["scale"], 4.0)
        self.assertEqual(payload["color_fix"], "none")
        self.assertEqual(payload["seed"], 0)

    def test_seed_upper_bound(self):
        payload = recavsr_pass.normalize_settings({"seed": 2**40})
        self.assertEqual(payload["seed"], recavsr_pass.SEED_MAX)

    def test_disabled_string_is_false(self):
        self.assertFalse(recavsr_pass.normalize_settings({"enabled": "false"})["enabled"])


class Status(unittest.TestCase):
    def test_missing_node_reports_not_available(self):
        status = recavsr_pass.status()
        self.assertIn("node", status)
        self.assertIn("transformer", status)


class Apply(unittest.TestCase):
    def test_disabled_returns_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.mp4"
            source.write_bytes(b"video")
            result = recavsr_pass.apply_recavsr(
                source, {"enabled": False}, out_dir=tmp, filename_prefix="clip"
            )
            self.assertEqual(Path(result), source.resolve())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["clip.mp4"])

    def test_missing_node_names_the_install_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.mp4"
            source.write_bytes(b"video")
            with unittest.mock.patch.object(recavsr_pass, "_node_classes", return_value=(None, None)):
                with self.assertRaises(ValueError) as error:
                    recavsr_pass.apply_recavsr(
                        source, {"enabled": True}, out_dir=tmp, filename_prefix="clip"
                    )
            self.assertEqual(str(error.exception), "Install ReCaVSR first.")

    def test_missing_transformer_names_the_model_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.mp4"
            source.write_bytes(b"video")
            with unittest.mock.patch.object(recavsr_pass, "_node_classes", return_value=(object, object)), \
                    unittest.mock.patch.object(recavsr_pass, "transformer_names", return_value=[]):
                with self.assertRaises(ValueError) as error:
                    recavsr_pass.apply_recavsr(
                        source, {"enabled": True}, out_dir=tmp, filename_prefix="clip"
                    )
            self.assertIn("recavsr", str(error.exception))

    def test_stub_node_copies_to_the_recavsr_suffix(self):
        class Loader:
            @staticmethod
            def execute(transformer):
                return FakeNodeOutput("model")

        class Upscale:
            calls = []

            @classmethod
            def execute(cls, model, **kwargs):
                cls.calls.append((model, kwargs))
                produced = Path(tempfile.mkdtemp()) / "produced.mp4"
                produced.write_bytes(b"upscaled")
                return FakeNodeOutput(FakeVideo(produced))

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.mp4"
            source.write_bytes(b"video")

            class Disk:
                @staticmethod
                def _next_output_path(out_dir, prefix, extension):
                    return Path(out_dir) / f"{prefix}.{extension}"

            with unittest.mock.patch.object(recavsr_pass, "_node_classes", return_value=(Loader, Upscale)), \
                    unittest.mock.patch.object(
                        recavsr_pass, "transformer_names", return_value=["transformer.safetensors"]
                    ), \
                    unittest.mock.patch.object(recavsr_pass, "_disk", return_value=Disk), \
                    unittest.mock.patch.object(recavsr_pass, "_video_input", return_value="video-input"):
                result = recavsr_pass.apply_recavsr(
                    source,
                    {"enabled": True, "scale": 2.0, "color_fix": "adain", "seed": 7},
                    out_dir=tmp,
                    filename_prefix="clip",
                )

            target = Path(result)
            self.assertEqual(target.name, "clip_recavsr.mp4")
            self.assertEqual(target.read_bytes(), b"upscaled")
            self.assertTrue(source.is_file())
            self.assertEqual(Upscale.calls[0][1]["scale"], 2.0)
            self.assertEqual(Upscale.calls[0][1]["color_fix"], "adain")
            self.assertEqual(Upscale.calls[0][1]["seed"], 7)
            self.assertEqual(Upscale.calls[0][1]["video"], "video-input")
            self.assertEqual(Upscale.calls[0][1]["decoder"], "flashdecoder")


class Wiring(unittest.TestCase):
    def test_extender_signature_carries_the_four_widgets(self):
        tree = ast.parse((ROOT / "extender.py").read_text(encoding="utf-8"))
        extend = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "extend"
        )
        names = [argument.arg for argument in extend.args.args]
        for name in ("recavsr_enabled", "recavsr_scale", "recavsr_color_fix", "recavsr_seed"):
            self.assertIn(name, names)

    def test_project_settings_record_the_mode(self):
        tree = ast.parse((ROOT / "extender.py").read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "PROJECT_SETTING_NAMES" for target in node.targets)
        )
        names = {element.value for element in assignment.value.elts}
        for name in ("recavsr_enabled", "recavsr_scale", "recavsr_color_fix", "recavsr_seed"):
            self.assertIn(name, names)

    def test_every_export_path_writes_the_mode(self):
        sources = {
            "extender.py": 2,
            "ref2va_independent.py": 1,
        }
        for filename, expected in sources.items():
            text = (ROOT / filename).read_text(encoding="utf-8")
            self.assertEqual(text.count('manifest["recavsr_mode"]'), expected, filename)

    def test_final_export_hook_exists(self):
        disk = (ROOT / "motion_context_disk.py").read_text(encoding="utf-8")
        engine = (ROOT / "fl2va_engine.py").read_text(encoding="utf-8")
        self.assertIn("_apply_recavsr(", disk)
        self.assertIn("_apply_recavsr(", engine)


if __name__ == "__main__":
    unittest.main()
