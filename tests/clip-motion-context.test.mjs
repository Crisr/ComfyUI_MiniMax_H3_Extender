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

test("new clips default Motion Context ON, except the first card", () => {
    const context = makeContext();
    assert.equal(vm.runInContext("newClip(0).motion_context", context), false);
    assert.equal(vm.runInContext("newClip(1).motion_context", context), true);
    assert.deepEqual(
        Array.from(vm.runInContext("[newClip(0), newClip(1)]", context), (c) => c.motion_context),
        [false, true],
    );
});

test("normalizeClipList keeps the per-clip flag and pins card 1 to OFF", () => {
    const context = makeContext();
    const clips = vm.runInContext(`normalizeClipList([
        { id: "a", motion_context: true },
        { id: "b", motion_context: false },
        { id: "c" },
        { id: "d", motion_context: true },
    ])`, context);
    assert.deepEqual(Array.from(clips, (c) => c.motion_context), [false, false, true, true]);
    // Pre-existing workflows have no key at all and must stay on Motion Context ON.
    const legacy = vm.runInContext('normalizeClipList([{ id: "x" }, { id: "y" }])', context);
    assert.deepEqual(Array.from(legacy, (c) => c.motion_context), [false, true]);
});

test("serialized workflow keeps the per-clip flag", () => {
    const context = makeContext();
    context.runtime = {
        state: {
            generation_mode: "ref2va",
            motion_context: true,
            clips: vm.runInContext('normalizeClipList([{ id: "a" }, { id: "b", motion_context: false }])', context),
        },
    };
    const json = vm.runInContext("serializeState(runtime.state)", context);
    assert.deepEqual(JSON.parse(json).clips.map((c) => c.motion_context), [false, false]);});
