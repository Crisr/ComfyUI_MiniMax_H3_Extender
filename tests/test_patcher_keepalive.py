"""CPU-only tests of patcher_keepalive; no ComfyUI or GPU required."""

import unittest

import patcher_keepalive


class _Module:
    pass


class _Patcher:
    def __init__(self, model):
        self.model = model


class _Entry:
    def __init__(self, patcher):
        self._patcher = patcher

    @property
    def model(self):
        return self._patcher


class _Manager:
    def __init__(self, entries=()):
        self.current_loaded_models = list(entries)


class KeepLoadedPatcherTest(unittest.TestCase):
    def setUp(self):
        patcher_keepalive.reset()
        self.module = _Module()

    def keep(self, patcher, entries=()):
        return patcher_keepalive.keep_loaded_patcher(
            patcher, manager=_Manager(entries)
        )

    def test_keeps_the_sampled_patcher(self):
        patcher = _Patcher(self.module)
        self.assertEqual([patcher], self.keep(patcher))

    def test_keeps_the_patcher_of_the_loaded_entry(self):
        delegate = _Patcher(self.module)
        sampled = _Patcher(self.module)
        kept = self.keep(sampled, [_Entry(delegate)])
        self.assertIn(delegate, kept)
        self.assertIn(sampled, kept)

    def test_ignores_entries_of_other_models(self):
        other = _Patcher(_Module())
        sampled = _Patcher(self.module)
        self.assertEqual([sampled], self.keep(sampled, [_Entry(other)]))

    def test_ignores_dead_entries(self):
        sampled = _Patcher(self.module)
        self.assertEqual([sampled], self.keep(sampled, [_Entry(None)]))

    def test_previous_patcher_survives_the_next_keep(self):
        first = _Patcher(self.module)
        second = _Patcher(self.module)
        self.keep(first)
        kept = self.keep(second, [_Entry(first)])
        self.assertIn(first, kept)
        self.assertIn(second, kept)
        self.assertEqual(patcher_keepalive.KEEP_LIMIT, len(kept))

    def test_keeps_only_the_limit(self):
        patchers = [_Patcher(self.module) for _ in range(4)]
        for patcher in patchers:
            kept = self.keep(patcher)
        self.assertEqual(patcher_keepalive.KEEP_LIMIT, len(kept))
        self.assertEqual(patchers[-patcher_keepalive.KEEP_LIMIT:], kept)

    def test_ignores_a_missing_model(self):
        self.assertEqual([], self.keep(None))

    def test_ignores_a_patcher_without_a_model(self):
        self.assertEqual([], self.keep(_Patcher(None)))


if __name__ == "__main__":
    unittest.main()
