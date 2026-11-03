import math
import unittest

import torch

from context_protection import (
    ProtectionSettings,
    build_context_noise_mask,
    build_masked_context,
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


class MaskedContextTests(unittest.TestCase):
    def test_masked_context_replaces_only_the_held_prefix(self):
        target_video = torch.arange(1 * 2 * 6 * 4 * 4).reshape(1, 2, 6, 4, 4).float()
        target_audio = torch.zeros(1, 2, 2, 30)
        held_video = target_video[:, :, :3].clone() + 100
        held_audio = torch.ones(1, 2, 2, 15)

        ctx = build_masked_context(
            target_video, target_audio, held_video, held_audio,
            noise_strength=0.5, noise_ease=False, trim_frames=9,
        )

        self.assertTrue(torch.equal(ctx.video[:, :, :3], held_video))
        self.assertTrue(torch.equal(ctx.video[:, :, 3:], target_video[:, :, 3:]))
        self.assertTrue(torch.equal(ctx.audio[..., :15], held_audio))
        self.assertTrue(torch.equal(ctx.audio[..., 15:], target_audio[..., 15:]))
        self.assertEqual(ctx.video_mask.shape, (1, 1, 6, 4, 4))
        self.assertEqual(ctx.video_mask[0, 0, :, 0, 0].tolist(), [0.5, 0.25, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ctx.audio_mask.shape, (1, 1, 1, 30))
        self.assertEqual(float(ctx.audio_mask.abs().sum()), 0.0)
        self.assertEqual(ctx.trim_frames, 9)

    def test_masked_context_rejects_geometry_mismatch(self):
        with self.assertRaises(ValueError):
            build_masked_context(
                torch.zeros(1, 2, 6, 4, 4), torch.zeros(1, 2, 2, 30),
                torch.zeros(1, 2, 3, 8, 8), torch.zeros(1, 2, 2, 15),
            )

    def test_masked_context_rejects_an_empty_held_context(self):
        with self.assertRaises(ValueError):
            build_masked_context(
                torch.zeros(1, 2, 6, 4, 4), torch.zeros(1, 2, 2, 30),
                torch.zeros(1, 2, 0, 4, 4), torch.zeros(1, 2, 2, 0),
            )
