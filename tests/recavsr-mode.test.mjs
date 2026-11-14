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

test("turning RECAVSR on turns REFINE off", () => {
    const context = makeContext();
    const next = vm.runInContext(
        'toggleUpscaleMode({ refine_enabled: true, recavsr_enabled: false }, "recavsr")',
        context,
    );
    assert.equal(next.recavsr_enabled, true);
    assert.equal(next.refine_enabled, false);
});

test("turning REFINE on turns RECAVSR off", () => {
    const context = makeContext();
    const next = vm.runInContext(
        'toggleUpscaleMode({ refine_enabled: false, recavsr_enabled: true }, "refine")',
        context,
    );
    assert.equal(next.refine_enabled, true);
    assert.equal(next.recavsr_enabled, false);
});

test("toggling RECAVSR off leaves REFINE untouched", () => {
    const context = makeContext();
    const next = vm.runInContext(
        'toggleUpscaleMode({ refine_enabled: false, recavsr_enabled: true }, "recavsr")',
        context,
    );
    assert.equal(next.recavsr_enabled, false);
    assert.equal(next.refine_enabled, false);
});

test("the two upscalers never share one settings-panel height", () => {
    const context = makeContext();
    const extra = (state) => vm.runInContext(`upscalePanelExtra(${JSON.stringify(state)})`, context);
    assert.equal(extra({ refine_enabled: false, recavsr_enabled: false }), 0);
    assert.equal(extra({ refine_enabled: true, recavsr_enabled: false }), 66);
    assert.equal(extra({ refine_enabled: false, recavsr_enabled: true }), 66);
    assert.equal(extra({ refine_enabled: true, recavsr_enabled: true }), 66);
});

test("the node check reads the LiteGraph registry", () => {
    const context = makeContext();
    assert.equal(
        vm.runInContext('recavsrNodeAvailable({ ReCaVSRUpscale: {} })', context),
        true,
    );
    assert.equal(vm.runInContext("recavsrNodeAvailable({})", context), false);
    assert.equal(vm.runInContext("recavsrNodeAvailable(null)", context), false);
});

test("RECAVSR adds one settings panel to the widget minimum height", () => {
    const context = makeContext();
    context.run = (state) => vm.runInContext(`uiMinHeightForState(${JSON.stringify(state)})`, context);
    const off = context.run({ generation_mode: "ref2va", motion_context: true, clips: [], refine_enabled: false, recavsr_enabled: false });
    const on = context.run({ generation_mode: "ref2va", motion_context: true, clips: [], refine_enabled: false, recavsr_enabled: true });
    assert.equal(on - off, 66);
});
