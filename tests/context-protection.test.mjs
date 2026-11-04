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
