import math
import unittest

import torch

from context_protection import (
    ProtectionSettings,
    build_context_noise_mask,
    capstone_index,
    normalize_protection_settings,
    protection_signature,
    resolve_sampler_step,
)


class ContextProtectionCoreTests(unittest.TestCase):
    def test_settings_are_normalized(self):
        settings = normalize_protection_settings(
            enabled=1, keystone=0, capstone=True,
            late_step=-4, noise_strength=1.7, noise_ease=0,
        )
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.keystone)
        self.assertTrue(settings.capstone)
        self.assertEqual(settings.late_step, 0)
        self.assertEqual(settings.noise_strength, 1.0)
        self.assertFalse(settings.noise_ease)

    def test_capstone_index_uses_end_aligned_h3_block(self):
        self.assertEqual(capstone_index(22), 20)
        self.assertEqual(capstone_index(25), 20)
        self.assertEqual(capstone_index(17), 15)
        self.assertEqual(capstone_index(1), 0)

    def test_noise_ramp_linear_and_eased(self):
        linear = build_context_noise_mask(5, 0.5, False)[0, 0, :, 0, 0].tolist()
        eased = build_context_noise_mask(5, 0.5, True)[0, 0, :, 0, 0].tolist()
        self.assertEqual(linear, [0.5, 0.375, 0.25, 0.125, 0.0])
        self.assertEqual(eased, [0.5, 0.28125, 0.125, 0.03125, 0.0])

    def test_single_context_step_uses_full_strength(self):
        values = build_context_noise_mask(1, 0.5, True)[0, 0, :, 0, 0].tolist()
        self.assertEqual(values, [0.5])

    def test_signature_is_off_without_an_active_source(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(
            protection_signature(None, settings, sigmas, "22", 0),
            "off",
        )
        self.assertEqual(
            protection_signature("clip_1", ProtectionSettings(False, True, True, 1, 0.5, True), sigmas, "22", 0),
            "off",
        )

    def test_signature_changes_with_each_effective_input(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        base = protection_signature("clip_1", settings, sigmas, "22", 0)
        self.assertNotEqual(base, protection_signature("clip_2", settings, sigmas, "22", 0))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "17", 0))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "22", 1))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "22", 0, 1.5))

    def test_resolve_sampler_step_uses_nearest_sigma(self):
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(resolve_sampler_step(1.0, sigmas), 0)
        self.assertEqual(resolve_sampler_step(0.49, sigmas), 1)
        self.assertEqual(resolve_sampler_step(0.0, sigmas), 2)
