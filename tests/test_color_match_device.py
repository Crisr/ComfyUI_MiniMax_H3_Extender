"""Tests for the device the Image Color Match runs on.

The node exposes no device option: the match must always use the default GPU,
exactly like the essentials Image Color Match with "gpu" selected. A silent
fall back to the CPU is what turns a 1-2 second step into a 200 second step.
"""

import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_color_match import HELPER_NAMES, _load  # noqa: E402

DEVICE_NAMES = HELPER_NAMES | {"_COLOR_MATCH_DEVICE"}


class ColorMatchDevice(unittest.TestCase):
    def setUp(self):
        self.ns = _load(["motion_context_disk.py"], DEVICE_NAMES)
        self._comfy = sys.modules.get("comfy")
        self._model_management = sys.modules.get("comfy.model_management")
        self.addCleanup(self._restore_modules)

    def _restore_modules(self):
        for name, module in (
            ("comfy", self._comfy),
            ("comfy.model_management", self._model_management),
        ):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def install_comfy(self, device=None, error=None):
        """Stand in for ComfyUI's device resolver."""
        module = types.ModuleType("comfy.model_management")

        def get_torch_device():
            if error is not None:
                raise error
            return device

        module.get_torch_device = get_torch_device
        package = types.ModuleType("comfy")
        package.model_management = module
        sys.modules["comfy"] = package
        sys.modules["comfy.model_management"] = module

    def helper(self, name):
        func = self.ns.get(name)
        self.assertIsNotNone(func, f"{name} is missing from motion_context_disk.py")
        return func

    def resolve(self):
        return torch.device(self.helper("_color_match_device")())

    def test_the_match_uses_the_comfyui_accelerator(self):
        self.install_comfy(torch.device("cuda:7"))
        self.assertEqual(self.resolve(), torch.device("cuda:7"))

    def test_a_cpu_report_from_comfyui_never_beats_an_available_gpu(self):
        self.install_comfy(torch.device("cpu"))
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(
            self.resolve().type,
            expected,
            "a CPU device must not win over an available GPU",
        )

    def test_a_missing_comfyui_still_resolves_a_device(self):
        self.install_comfy(error=RuntimeError("comfy is not available"))
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(self.resolve().type, expected)

    def test_the_device_is_resolved_once(self):
        self.install_comfy(torch.device("cuda:1"))
        resolve = self.helper("_color_match_device")
        self.assertIs(resolve(), resolve())


if __name__ == "__main__":
    unittest.main()
