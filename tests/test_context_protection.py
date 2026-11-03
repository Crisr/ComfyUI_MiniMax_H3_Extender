import math
import unittest

import torch

from context_protection import (
    ProtectionSettings,
    build_context_noise_mask,
    build_masked_context,
    capstone_index,
    make_protection_plan,
    make_protection_wrapper,
    normalize_protection_settings,
    normalize_protection_source,
    protection_cache_mismatch,
    protection_signature,
    resolve_sampler_step,
    resolve_source_index,
    restore_samples,
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


class ProtectionPlanTests(unittest.TestCase):
    @staticmethod
    def _fake_masked():
        target_video = torch.zeros(1, 2, 6, 4, 4)
        held_video = torch.ones(1, 2, 3, 4, 4)
        return build_masked_context(
            target_video, torch.zeros(1, 2, 2, 30),
            held_video, torch.ones(1, 2, 2, 15),
            noise_strength=0.5, noise_ease=False, trim_frames=9,
        )

    @staticmethod
    def _plan_with_held_values():
        settings = ProtectionSettings(True, False, True, 1, 0.5, True)
        return make_protection_plan(
            ProtectionPlanTests._fake_masked(),
            torch.zeros(1, 2, 22, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )

    def test_plan_rejects_late_step_outside_schedule(self):
        settings = ProtectionSettings(True, True, True, 3, 0.5, True)
        masked = self._fake_masked()
        with self.assertRaises(ValueError):
            make_protection_plan(
                masked, torch.zeros(1, 2, 22, 4, 4), "clip_1", 0, settings,
                torch.tensor([1.0, 0.5, 0.0]), "22", 0,
            )

    def test_plan_skips_capstone_for_short_source(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        plan = make_protection_plan(
            self._fake_masked(), torch.zeros(1, 2, 5, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )
        self.assertIsNone(plan.capstone)

    def test_protection_wrapper_delays_front_mask_before_late_step(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        plan = make_protection_plan(
            self._fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )
        wrapper = make_protection_wrapper(plan)
        seen = {}

        def executor(x, timestep, context, transformer_options, **kwargs):
            seen["t"] = x[0].shape[2]
            seen["front"] = kwargs["denoise_mask"][0, 0, :3, 0, 0].tolist()
            seen["audio_front"] = kwargs["audio_denoise_mask"][0, 0, 0, :3].tolist()
            return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

        x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
        mask = torch.empty(1, 1, 22, 4, 4).fill_(0.5)
        audio_mask = torch.zeros(1, 1, 1, 100)
        wrapper(executor, x, torch.tensor(1000.0), None,
                {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
                denoise_mask=mask, audio_denoise_mask=audio_mask)
        self.assertEqual(seen["t"], 22)
        self.assertEqual(seen["front"], [1.0, 1.0, 1.0])
        self.assertEqual(seen["audio_front"], [1.0, 1.0, 1.0])

    def test_protection_wrapper_appends_then_trims_after_late_step(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        plan = make_protection_plan(
            self._fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )
        wrapper = make_protection_wrapper(plan)
        seen = {}

        def executor(x, timestep, context, transformer_options, **kwargs):
            seen["t"] = x[0].shape[2]
            seen["mask_t"] = kwargs["denoise_mask"].shape[2]
            seen["front"] = kwargs["denoise_mask"][0, 0, :3, 0, 0].tolist()
            return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

        x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
        mask = torch.ones(1, 1, 22, 4, 4)
        out = wrapper(executor, x, torch.tensor(500.0), None, {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
                      denoise_mask=mask, audio_denoise_mask=torch.ones(1, 1, 1, 100))
        self.assertEqual(seen["t"], 27)
        self.assertEqual(seen["mask_t"], 27)
        self.assertEqual(seen["front"], [0.5, 0.25, 0.0])
        self.assertEqual(out[0].shape[2], 22)

    def test_capstone_wrapper_cleans_up_when_executor_raises(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        plan = make_protection_plan(
            self._fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )
        wrapper = make_protection_wrapper(plan)
        x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
        mask = torch.ones(1, 1, 22, 4, 4)

        def executor(*args, **kwargs):
            raise RuntimeError("sampler failed")

        with self.assertRaises(RuntimeError):
            wrapper(executor, x, torch.tensor(500.0), None,
                    {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
                    denoise_mask=mask,
                    audio_denoise_mask=torch.ones(1, 1, 1, 100))

        self.assertEqual(x[0].shape[2], 22)
        self.assertEqual(mask.shape[2], 22)

    def test_restore_samples_replaces_the_held_prefix(self):
        plan = self._plan_with_held_values()
        video = torch.full((1, 2, 6, 4, 4), 9.0)
        audio = torch.full((1, 2, 2, 30), 9.0)
        restored = restore_samples((video, audio), plan)
        self.assertTrue(torch.equal(restored[0][:, :, :3], plan.masked.held_video))
        self.assertTrue(torch.equal(restored[0][:, :, 3:], video[:, :, 3:]))
        self.assertEqual(float(restored[0][:, :, :3].min()), 1.0)


class SourceResolutionTests(unittest.TestCase):
    def test_source_resolution_rejects_missing_current_and_future(self):
        clips = [{"id": "clip_a"}, {"id": "clip_b"}, {"id": "clip_c"}]
        self.assertEqual(resolve_source_index(clips, 2, "clip_a"), 0)
        for bad in ("missing", "clip_b", "clip_c"):
            with self.assertRaises(ValueError):
                resolve_source_index(clips, 1, bad)

    def test_missing_source_defaults_to_first_clip_but_explicit_null_stays_null(self):
        self.assertEqual(normalize_protection_source({}, 2, "clip_a"), "clip_a")
        self.assertEqual(normalize_protection_source({"protection_source": ""}, 2, "clip_a"), "clip_a")
        self.assertEqual(normalize_protection_source({"protection_source": "clip_b"}, 2, "clip_a"), "clip_b")
        self.assertIsNone(normalize_protection_source({"protection_source": None}, 2, "clip_a"))
        self.assertIsNone(normalize_protection_source({}, 0, "clip_a"))

    def test_cache_mismatch_treats_missing_signature_as_off(self):
        self.assertFalse(protection_cache_mismatch({}, "off"))
        self.assertTrue(protection_cache_mismatch({}, "abc"))
        self.assertFalse(protection_cache_mismatch({"protection_signature": "abc"}, "abc"))
        self.assertTrue(protection_cache_mismatch({"protection_signature": "abc"}, "off"))
