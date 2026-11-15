import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
    process.env.H3_EXTENDER_SOURCE || new URL("../web/extender.js", import.meta.url),
    "utf8",
).replace(/^import .*;\r?\n/gm, "");

function makeContext() {
    const context = vm.createContext({
        app: { registerExtension() {}, graph: null },
        api: { apiURL(value) { return value; } },
        document: {
            getElementById() { return {}; },
            createElement() { return { dataset: {}, style: { setProperty() {} }, append() {}, addEventListener() {} }; },
        },
        requestAnimationFrame() {},
        alert() {},
        console,
        Set,
    });
    vm.runInContext(source, context);
    return context;
}

test("new clips start with no explicit protection source", () => {
    const context = makeContext();
    assert.equal(vm.runInContext("newClip(0).protection_source", context), null);
    assert.equal(vm.runInContext("newClip(1).protection_source", context), null);
});

test("normalization defaults a missing source to the first clip", () => {
    const context = makeContext();
    const clips = vm.runInContext(`normalizeClipList([
        { id: "a" },
        { id: "b" },
        { id: "c", protection_source: null },
    ])`, context);
    assert.equal(clips[0].protection_source, null);
    assert.equal(clips[1].protection_source, "a");
    assert.equal(clips[2].protection_source, null);
});

test("options expose only earlier clips plus None", () => {
    const context = makeContext();
    const options = vm.runInContext(
        'protectionSourceOptions([{ id: "a" }, { id: "b" }, { id: "c" }], 2)',
        context,
    );
    assert.deepEqual(Array.from(options, (item) => item.value), ["", "a", "b"]);
});

test("protection widgets ship the opt-in defaults", () => {
    const context = makeContext();
    const defaults = vm.runInContext("PROTECTION_WIDGET_DEFAULTS", context);
    assert.equal(defaults.protection_noise_fade, "auto");
    assert.equal(defaults.protection_audio_feather, 0);
    assert.equal(defaults.protection_late_lock_slots, "both");
    const widgets = Array.from(vm.runInContext("PROJECT_WIDGETS", context));
    for (const name of [
        "protection_noise_fade",
        "protection_audio_feather",
        "protection_late_lock_slots",
    ]) {
        assert.ok(widgets.includes(name), `${name} missing from PROJECT_WIDGETS`);
    }
});

test("serialized workflow keeps the per-clip protection source", () => {
    const context = makeContext();
    context.runtime = {
        state: {
            generation_mode: "ref2va",
            motion_context: true,
            clips: vm.runInContext(`normalizeClipList([
                { id: "a" },
                { id: "b" },
                { id: "c", protection_source: null },
            ])`, context),
        },
    };
    const json = vm.runInContext("serializeState(runtime.state)", context);
    assert.deepEqual(
        JSON.parse(json).clips.map((clip) => clip.protection_source),
        [null, "a", null],
    );
});

test("invalid restored widget values fall back to their declared defaults", () => {
    const context = makeContext();
    const repair = (name, value) =>
        vm.runInContext(`repairedWidgetValue(${JSON.stringify(name)}, ${JSON.stringify(value)})`, context);
    // The exact shift a stale positional restore produced in the field.
    assert.equal(repair("protection_noise_fade", "learned"), "auto");
    assert.equal(repair("protection_noise_fade", ""), "auto");
    assert.equal(repair("protection_audio_feather", "both"), 0);
    assert.equal(repair("protection_late_lock_slots", "learned"), "both");
    // Valid values pass through, and numbers are clamped, not replaced.
    assert.equal(repair("protection_noise_fade", "ethanfel"), "ethanfel");
    assert.equal(repair("protection_audio_feather", "8"), 8);
    assert.equal(repair("protection_audio_feather", 99), 64);
    assert.equal(repair("protection_noise_strength", "0.75"), 0.75);
    assert.equal(repair("protection_noise_strength", 2), 1);
    assert.equal(repair("protection_late_step", "abc"), 1);
    assert.equal(repair("refine_upscaler", "learned"), "learned");
    assert.equal(repair("refine_upscaler", "nearest"), "learned");
});

test("an empty restored widget value is reset instead of skipped", () => {
    const context = makeContext();
    const node = {
        widgets: [
            { name: "protection_noise_fade", value: "" },
            { name: "protection_audio_feather", value: "" },
            { name: "protection_late_lock_slots", value: "" },
        ],
    };
    context.__node = node;
    vm.runInContext("repairProtectionWidgetValues(__node)", context);
    assert.equal(node.widgets[0].value, "auto");
    assert.equal(node.widgets[1].value, 0);
    assert.equal(node.widgets[2].value, "both");
});
