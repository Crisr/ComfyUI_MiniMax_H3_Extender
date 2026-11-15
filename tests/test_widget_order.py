"""Guard the positional widget order of the Extender node.

ComfyUI restores a saved workflow's widget values by position. A widget that
lands in the middle of ``INPUT_TYPES["required"]`` therefore shifts every older
value onto the wrong name: a saved ``refine_upscaler`` value reached
``protection_noise_fade`` in the field. New widgets belong at the very end.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "extender.py"
NODE_CLASS = "MiniMaxH3Extender"

# The tail added by v3.2.0 and the widget it must follow.
LAST_LEGACY_WIDGET = "refine_upscaler"
APPENDED_WIDGETS = (
    "protection_noise_fade",
    "protection_audio_feather",
    "protection_late_lock_slots",
)


def required_widget_order(source: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == NODE_CLASS:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                    return _required_keys(item)
    raise AssertionError(f"{NODE_CLASS}.INPUT_TYPES was not found")


def _required_keys(function: ast.FunctionDef) -> list[str]:
    assignments = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    for node in ast.walk(function):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "required":
                if isinstance(value, ast.Name):
                    value = assignments.get(value.id)
                if not isinstance(value, ast.Dict):
                    raise AssertionError("required is not a dict literal")
                keys = []
                for entry in value.keys:
                    if not isinstance(entry, ast.Constant):
                        raise AssertionError("required holds a non-literal key")
                    keys.append(str(entry.value))
                return keys
    raise AssertionError("INPUT_TYPES does not return a required dict")


class WidgetOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.order = required_widget_order(SOURCE.read_text(encoding="utf-8"))

    def test_appended_widgets_are_the_last_ones(self):
        tail = self.order[-(len(APPENDED_WIDGETS) + 1):]
        self.assertEqual(tail, [LAST_LEGACY_WIDGET, *APPENDED_WIDGETS])

    def test_no_widget_repeats(self):
        self.assertEqual(len(self.order), len(set(self.order)))

    def test_widget_order_covers_every_protection_setting(self):
        for name in APPENDED_WIDGETS:
            self.assertIn(name, self.order)
            self.assertGreater(
                self.order.index(name),
                self.order.index(LAST_LEGACY_WIDGET),
                f"{name} must stay behind {LAST_LEGACY_WIDGET}",
            )


if __name__ == "__main__":
    unittest.main()
