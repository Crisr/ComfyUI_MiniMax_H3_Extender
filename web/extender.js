import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3Extender";
const FINAL_TARGET = "MiniMaxH3MotionContextDiskFinalDecode";
const PROGRESS_EVENT = "h3_extender_progress";
const PROMPT_PACK_EVENT = "h3_extender_prompt_pack_import";
const REF_PACK_EVENT = "h3_extender_ref_pack_import";
const CARD_WIDTH = 318;
const UI_MIN_HEIGHT = 600;
const NODES2_MIN_HEIGHT = 650;
// Keep a real visual gap between the native Nodes 2.0 widgets and the CLIP
// panel. This is internal padding only: we deliberately do NOT rewrite Vue
// grid tracks or absolutely position the DOM widget.
const NODES2_TOP_GAP = 28;
const NODE_MIN_WIDTH = 520;
const BOTTOM_PAD = 16;
// Leave an empty gutter under each card so an overlay horizontal scrollbar
// never covers the Validated/footer row.
const CARD_SCROLLBAR_SPACE = 24;
const CARD_MIN_HEIGHT = 355;
const NODES2_CARDS_MIN_HEIGHT = CARD_MIN_HEIGHT + CARD_SCROLLBAR_SPACE;
const REF_SLOT_WIDTH = 96;
const REF_THUMB_HEIGHT = 96;
// Reserve the scrollbar inside the existing reference section only.
// Do not grow the DOM widget or alter card sizing/layout for this.
const REF_SCROLLBAR_SPACE = 14;
const REF_SECTION_HEIGHT = 160;
const MAX_IMAGE_REFS = 9;
const MAX_RESOLUTION = 4096;
const DEFAULT_MEGAPIXELS = 0.40;

const PROJECT_WIDGETS = [
    "run_mode",
    "width",
    "height",
    "ref_image_size",
    "steps",
    "sampler_name",
    "scheduler",
    "denoise",
    "context_length",
    "audio_context_length",
    "clips_json",
    "resolution_mode",
    "megapixels",
    "refs_json",
];

const FINAL_PROJECT_WIDGETS = [
    "fps",
    "filename_prefix",
    "output_directory",
    "codec",
    "crf",
    "preset",
    "audio_bitrate",
];

// Validation and reference semantics are user-controlled. The Extender never
// associates Ref N with Clip N and never decides which clip a reference edit
// invalidates. Existing validation flags stay exactly as the user left them.
// The one unavoidable global exception is RESOLUTION: cached latents cannot be
// reused at another geometry, so an effective width/height change immediately
// clears validation for the whole chain.


function emptyRefsState() {
    return { version: 2, refs: Array(MAX_IMAGE_REFS).fill(null) };
}

function normalizeRefDescriptor(value) {
    if (!value || typeof value !== "object") return null;
    const id = String(value.id || value.ref_id || "").toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(id)) return null;
    const sourceCandidate = String(value.source_id || value.original_id || id).toLowerCase();
    const source_id = /^[0-9a-f]{64}$/.test(sourceCandidate) ? sourceCandidate : id;
    const adjustment = (name) => {
        const n = Number(value[name] ?? 100);
        return Number.isFinite(n) ? Math.min(200, Math.max(0, n)) : 100;
    };
    const descriptor = {
        id,
        source_id,
        original_name: String(value.original_name || value.name || "reference.png"),
        width: Math.max(0, Number(value.width || 0)),
        height: Math.max(0, Number(value.height || 0)),
        size_bytes: Math.max(0, Number(value.size_bytes || 0)),
        saturation: adjustment("saturation"),
        contrast: adjustment("contrast"),
        brightness: adjustment("brightness"),
    };
    const externalSignature = String(value.external_signature || "").toLowerCase();
    if (/^[0-9a-f]{64}$/.test(externalSignature)) {
        descriptor.external_signature = externalSignature;
    }
    return descriptor;
}

function normalizeRefsArray(values) {
    // Ref slots are stable logical identities. Never compact holes: moving Ref 3
    // into Ref 2 would silently break prompts that intentionally use <Picture 3>.
    const refs = Array(MAX_IMAGE_REFS).fill(null);
    const source = Array.isArray(values) ? values : [];
    for (let i = 0; i < Math.min(MAX_IMAGE_REFS, source.length); i++) {
        refs[i] = normalizeRefDescriptor(source[i]);
    }
    return refs;
}

function parseRefsState(raw) {
    try {
        const parsed = typeof raw === "string" ? JSON.parse(raw || "{}") : raw;
        const refs = Array.isArray(parsed) ? parsed : parsed?.refs;
        return { version: 2, refs: normalizeRefsArray(Array.isArray(refs) ? refs : []) };
    } catch (_) {
        return emptyRefsState();
    }
}

function serializeRefsState(state) {
    return JSON.stringify({ version: 2, refs: normalizeRefsArray(state?.refs || []) });
}

function refCount(runtime) {
    return (runtime?.refsState?.refs || []).filter(Boolean).length;
}

function refImageUrl(ref) {
    if (!ref?.id) return "";
    return api.apiURL("/h3_extender/ref/image?id=" + encodeURIComponent(String(ref.id)));
}

function sameRefContent(a, b) {
    return String(a?.id || "") === String(b?.id || "");
}

function removeLegacyImageRefInputs(node) {
    if (!node?.inputs) return false;
    let removed = false;
    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const name = String(node.inputs[index]?.name || "");
        if (/^ref_[1-9]$/.test(name)) {
            try {
                node.removeInput(index);
                removed = true;
            } catch (_) {}
        }
    }
    if (removed) node.graph?.setDirtyCanvas(true, true);
    return removed;
}

function randomSeed() {
    try {
        const a = new Uint32Array(2);
        crypto.getRandomValues(a);
        // stay inside JS exact-integer range
        return Number((BigInt(a[0]) << 21n) ^ BigInt(a[1] & 0x1fffff));
    } catch (_) {
        return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    }
}

function normalizeColorAdjustment(value) {
    const c = value && typeof value === "object" ? value : {};
    const clamp = (v, lo, hi, fallback) => {
        const n = Number(v);
        return Math.max(lo, Math.min(hi, Number.isFinite(n) ? n : fallback));
    };
    return {
        saturation: clamp(c.saturation, 0, 200, 100),
        contrast: clamp(c.contrast, 50, 150, 100),
        brightness: clamp(c.brightness, 50, 150, 100),
    };
}

function colorAdjustmentIsNeutral(value) {
    const c = normalizeColorAdjustment(value);
    return [c.saturation, c.contrast, c.brightness].every((v) => Math.abs(v - 100) < 1e-6);
}

function cssColorFilter(value) {
    const c = normalizeColorAdjustment(value);
    return `saturate(${c.saturation}%) contrast(${c.contrast}%) brightness(${c.brightness}%)`;
}

function newClip(index) {
    return {
        id: `clip_${index + 1}_${Date.now().toString(36)}`,
        name: "",
        prompt: "",
        seed: randomSeed(),
        seed_mode: "randomize",
        duration: 10.0,
        validated: false,
        color_adjustment: normalizeColorAdjustment(),
    };
}

function parseState(raw) {
    try {
        const p = JSON.parse(raw || "{}");
        const clips = Array.isArray(p) ? p : p?.clips;
        if (Array.isArray(clips) && clips.length) {
            return {
                version: 1,
                load_token: String(p?.project_load_token || ""),
                prompt_pack_signature: String(p?.prompt_pack_signature || ""),
                clips: clips.map((c, i) => ({
                    id: String(c?.id || `clip_${i + 1}`),
                    name: String(c?.name || ""),
                    prompt: String(c?.prompt || ""),
                    seed: Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Number(c?.seed || 0))),
                    seed_mode: ["randomize", "fixed", "increment", "decrement"].includes(String(c?.seed_mode))
                        ? String(c.seed_mode)
                        : "randomize",
                    duration: Math.max(0.25, Math.min(150, Number(c?.duration || 10))),
                    validated: Boolean(c?.validated),
                    color_adjustment: normalizeColorAdjustment(c?.color_adjustment),
                })),
            };
        }
    } catch (_) {}
    return { version: 1, load_token: "", prompt_pack_signature: "", clips: [newClip(0)] };
}

function serializeState(state) {
    const payload = { version: 1, clips: state.clips };
    if (state?.load_token) payload.project_load_token = String(state.load_token);
    if (state?.prompt_pack_signature) {
        payload.prompt_pack_signature = String(state.prompt_pack_signature);
    }
    return JSON.stringify(payload);
}

function validatedPrefixFromState(state) {
    let count = 0;
    for (const clip of state?.clips || []) {
        if (!clip?.validated) break;
        count += 1;
    }
    return count;
}

async function restoreCacheState(node, runtime) {
    if (!node || !runtime || runtime.cacheStateRequestRunning) return;

    runtime.cacheStateRequestRunning = true;
    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(node.id));
        const response = await fetch(
            api.apiURL("/h3_extender/cache_state?" + params.toString())
        );
        if (!response.ok) return;

        const payload = await response.json();
        if (!payload?.found) return;

        // Do not overwrite live execution information if generation started
        // while the startup request was in flight.
        if (["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""))) {
            return;
        }

        runtime.cachedCount = Number(payload.cached_count || 0);
        runtime.validatedCount = Number(payload.validated_count || 0);
        const restoredW = Number(payload.resolved_width || 0);
        const restoredH = Number(payload.resolved_height || 0);
        if (restoredW > 0 && restoredH > 0) {
            // Cache restore is informational only. Do not overwrite live
            // resolution controls: outside an explicit .ext Load the user is
            // free to change Auto/MP or Manual width/height at any time.
            runtime.expectedResolution = { width: restoredW, height: restoredH };
        }
        runtime.cacheStateRestored = true;
        const resolutionText = restoredW > 0 && restoredH > 0
            ? ` | project ${restoredW}x${restoredH}`
            : "";
        runtime.statusText =
            `Restored cache${resolutionText} | cached ${runtime.cachedCount}/${runtime.state.clips.length} | ` +
            `validated ${runtime.validatedCount}`;
        syncResolutionAndInvalidate(node, runtime);
        render(node, runtime);
        node.graph?.setDirtyCanvas(true, true);
    } catch (_) {
        // Cache-state restoration is visual convenience only. Never block UI load.
    } finally {
        runtime.cacheStateRequestRunning = false;
    }
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function effectiveManualResolution(width, height) {
    const step = 32;
    const w = Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(Number(width || 0) / step) * step));
    const h = Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(Number(height || 0) / step) * step));
    return { width: w, height: h };
}

function pythonRound(value) {
    // Python round() uses bankers rounding for exact .5 ties; match the
    // backend so the visible mirror can never disagree by a latent-grid step.
    const x = Number(value);
    if (!Number.isFinite(x)) return 0;
    const floor = Math.floor(x);
    const frac = x - floor;
    if (Math.abs(frac - 0.5) < 1e-12) return (floor % 2 === 0) ? floor : floor + 1;
    return Math.round(x);
}

function autoResolutionFromDimensions(srcWidth, srcHeight, megapixels) {
    const srcW = Number(srcWidth || 0);
    const srcH = Number(srcHeight || 0);
    if (!(srcW > 0) || !(srcH > 0)) return null;

    const mp = Math.max(0.01, Math.min(16.0, Number(megapixels ?? DEFAULT_MEGAPIXELS)));
    const total = mp * 1024.0 * 1024.0;
    const scale = Math.sqrt(total / (srcW * srcH));
    let scaledW = srcW * scale;
    let scaledH = srcH * scale;

    if (scaledW > MAX_RESOLUTION || scaledH > MAX_RESOLUTION) {
        const shrink = Math.min(MAX_RESOLUTION / scaledW, MAX_RESOLUTION / scaledH);
        scaledW *= shrink;
        scaledH *= shrink;
    }

    // H3 32-pixel canvas. Auto snaps downward so the resolved canvas never
    // exceeds the requested megapixel budget; Manual uses the same 32px grid.
    const step = 32;
    return {
        width: Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(scaledW / step) * step)),
        height: Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(scaledH / step) * step)),
    };
}

function currentGuideRefNumber(runtime) {
    const refs = runtime?.refsState?.refs || [];
    if (refs[0]) return 1;
    for (let i = 0; i < Math.min(MAX_IMAGE_REFS, refs.length); i++) {
        if (refs[i]) return i + 1;
    }
    return null;
}

function dimensionsFromInternalRef(runtime, refNumber) {
    const index = Number(refNumber) - 1;
    if (!runtime || !Number.isInteger(index) || index < 0 || index >= MAX_IMAGE_REFS) return null;
    const ref = runtime.refsState?.refs?.[index];
    const width = Number(ref?.width || 0);
    const height = Number(ref?.height || 0);
    return width > 0 && height > 0 ? { width, height } : null;
}

function setResolutionMirrorValues(node, runtime, width, height) {
    if (!runtime || !(width > 0) || !(height > 0)) return;
    runtime.applyingResolutionMirror = true;
    try {
        setWidgetValue(node, "width", Number(width));
        setWidgetValue(node, "height", Number(height));
    } finally {
        runtime.applyingResolutionMirror = false;
    }
}

function rememberManualResolution(node, runtime, width, height) {
    if (!runtime) return;
    if (Number(width) > 0) runtime.manualWidth = Number(width);
    if (Number(height) > 0) runtime.manualHeight = Number(height);
    if (node) {
        node.properties = node.properties || {};
        if (runtime.manualWidth > 0) node.properties.h3_manual_width = runtime.manualWidth;
        if (runtime.manualHeight > 0) node.properties.h3_manual_height = runtime.manualHeight;
    }
}

function syncResolutionMirror(node, runtime) {
    if (!node || !runtime) return;

    const mode = String(getWidget(node, "resolution_mode")?.value || "auto_from_ref");
    const widthWidget = getWidget(node, "width");
    const heightWidget = getWidget(node, "height");
    if (!widthWidget || !heightWidget) return;

    if (mode === "manual") {
        if (runtime.manualWidth > 0 && runtime.manualHeight > 0) {
            setResolutionMirrorValues(node, runtime, runtime.manualWidth, runtime.manualHeight);
        }
        runtime.resolutionMirrorActive = false;
        return;
    }

    const guideRef = currentGuideRefNumber(runtime);
    if (guideRef == null) {
        // Auto without a reference is deliberately the editable Manual fallback.
        if (runtime.manualWidth > 0 && runtime.manualHeight > 0) {
            setResolutionMirrorValues(node, runtime, runtime.manualWidth, runtime.manualHeight);
        }
        runtime.resolutionMirrorActive = false;
        return;
    }

    let source = dimensionsFromInternalRef(runtime, guideRef);
    const executedGuide = /^ref_(\d+)$/.exec(String(runtime.resolutionGuide || ""));
    if (!source && executedGuide && Number(executedGuide[1]) === Number(guideRef)) {
        if (runtime.guideSourceWidth > 0 && runtime.guideSourceHeight > 0) {
            source = { width: runtime.guideSourceWidth, height: runtime.guideSourceHeight };
        }
    }

    if (!source) {
        // Internal metadata normally carries the exact source dimensions. Keep
        // the last backend result as a defensive fallback for older saved state.
        if (
            executedGuide &&
            Number(executedGuide[1]) === Number(guideRef) &&
            runtime.resolvedWidth > 0 && runtime.resolvedHeight > 0
        ) {
            setResolutionMirrorValues(node, runtime, runtime.resolvedWidth, runtime.resolvedHeight);
            runtime.resolutionMirrorActive = true;
        }
        return;
    }

    const resolved = autoResolutionFromDimensions(
        source.width,
        source.height,
        Number(getWidget(node, "megapixels")?.value ?? DEFAULT_MEGAPIXELS),
    );
    if (!resolved) return;
    runtime.guideSourceWidth = Number(source.width);
    runtime.guideSourceHeight = Number(source.height);
    setResolutionMirrorValues(node, runtime, resolved.width, resolved.height);
    runtime.resolutionMirrorActive = true;
    node.graph?.setDirtyCanvas(true, true);
}

function wrapResolutionWidgetCallbacks(node, runtime) {
    if (!node || !runtime || runtime.resolutionCallbacksInstalled) return;
    runtime.resolutionCallbacksInstalled = true;

    const widthWidget = getWidget(node, "width");
    const heightWidget = getWidget(node, "height");
    const modeWidget = getWidget(node, "resolution_mode");
    const mpWidget = getWidget(node, "megapixels");

    const wrap = (widget, handler) => {
        if (!widget) return;
        const old = widget.callback;
        widget.callback = function (value) {
            const result = old ? old.apply(this, arguments) : undefined;
            handler(value);
            return result;
        };
    };

    wrap(widthWidget, (value) => {
        if (runtime.applyingResolutionMirror) return;
        const mode = String(modeWidget?.value || "auto_from_ref");
        if (mode === "manual" || currentGuideRefNumber(runtime) == null) {
            // A loaded .ext is forced to its exact archived geometry only at
            // load time. The first explicit resolution edit releases that
            // one-shot project state immediately.
            runtime.projectResolutionLoaded = false;
            rememberManualResolution(
                node,
                runtime,
                Number(value || widthWidget?.value || runtime.manualWidth || 896),
                runtime.manualHeight,
            );
            invalidateForResolutionChange(node, runtime);
        } else {
            requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
        }
    });
    wrap(heightWidget, (value) => {
        if (runtime.applyingResolutionMirror) return;
        const mode = String(modeWidget?.value || "auto_from_ref");
        if (mode === "manual" || currentGuideRefNumber(runtime) == null) {
            runtime.projectResolutionLoaded = false;
            rememberManualResolution(
                node,
                runtime,
                runtime.manualWidth,
                Number(value || heightWidget?.value || runtime.manualHeight || 576),
            );
            invalidateForResolutionChange(node, runtime);
        } else {
            requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
        }
    });
    wrap(modeWidget, (value) => {
        runtime.projectResolutionLoaded = false;
        const mode = String(value || modeWidget?.value || "auto_from_ref");
        if (mode === "auto_from_ref" && !runtime.resolutionMirrorActive) {
            rememberManualResolution(
                node,
                runtime,
                Number(widthWidget?.value || runtime.manualWidth || 896),
                Number(heightWidget?.value || runtime.manualHeight || 576),
            );
        }
        requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
    });
    wrap(mpWidget, () => {
        // Load Project deliberately enters Manual at the archived width/height
        // so simply pressing Queue cannot invalidate the imported cache. But
        // megapixels is an Auto-only control: if the user edits it after a
        // project load, that is an explicit request to choose a new resolution.
        // Re-enter Auto immediately instead of leaving the MP widget apparently
        // ineffective. The backend will then restart the incompatible cache on
        // the next generation, exactly like any other live resolution change.
        if (runtime.projectResolutionLoaded) {
            runtime.projectResolutionLoaded = false;
            setWidgetValue(node, "resolution_mode", "auto_from_ref");
        }
        requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
    });
}

// Nodes 2.0 (Vue) can render the native multiline STRING row before our
// onNodeCreated code gets a chance to touch the widget object. Hide that row
// pre-emptively with CSS, using the same proven strategy as ComfyUI_Stem_Mixer.
// MiniMaxH3Extender keeps clips_json + refs_json as native serialized textareas.
(function injectStateJsonHideRule() {
    if (document.getElementById("h3-extender-hide-state-json")) return;
    const style = document.createElement("style");
    style.id = "h3-extender-hide-state-json";
    style.textContent = `
        .lg-node-widget:has(> [node-type="${TARGET}"] > textarea) {
            display: none !important;
        }
    `;
    document.head.appendChild(style);
})();

function hideNativeWidget(node, widget) {
    if (!widget) return;

    // LiteGraph / Nodes 1.0: remove the logical footprint but keep the widget
    // itself intact so its normal workflow serialization continues to work.
    // Do NOT use canvasOnly/hidden here: Vue does not reliably honour those
    // flags for native widgets, while the pre-emptive CSS above does.
    widget.computeSize = () => [0, -4];
    widget.computeLayoutSize = () => ({
        minWidth: 0,
        minHeight: 0,
        maxWidth: 0,
        maxHeight: 0,
    });

    // LiteGraph may recreate the textarea when the node leaves/re-enters the
    // viewport, so re-hide the actual legacy DOM element on every foreground
    // draw, exactly as Stem Mixer does for its state widget.
    const oldDrawForeground = node?.onDrawForeground;
    if (node) {
        node.onDrawForeground = function (ctx) {
            if (oldDrawForeground) oldDrawForeground.apply(this, arguments);
            const inputEl = widget.inputEl;
            if (inputEl) {
                if (inputEl.style.display !== "none") inputEl.style.display = "none";
                const parent = inputEl.parentElement;
                if (parent && parent.style.display !== "none") {
                    parent.style.display = "none";
                }
            }
        };
    }
}

function domWidgetRenderMode(element) {
    // ComfyUI exposes the renderer state on LiteGraph.vueNodesMode. Use that
    // as the authority, but wait while the DOM widget is being re-parented so
    // we never apply Legacy sizing with a stale Vue last_y (or vice versa).
    const LG = globalThis.LiteGraph;
    const hasModeFlag = typeof LG?.vueNodesMode === "boolean";
    if (!element?.isConnected) return "pending";

    const insideVueRow = Boolean(element.closest?.(".lg-node-widget"));
    if (hasModeFlag) {
        if (LG.vueNodesMode && !insideVueRow) return "pending";
        if (!LG.vueNodesMode && insideVueRow) return "pending";
        return LG.vueNodesMode ? "nodes2" : "legacy";
    }

    // Older frontends may not expose vueNodesMode; fall back to the wrapper.
    return insideVueRow ? "nodes2" : "legacy";
}

function obviouslyPoisonedHeight(height, minimumHeight) {
    const h = Number(height);
    if (!Number.isFinite(h) || h <= 0) return false;
    return h > Math.max(1800, Number(minimumHeight || 0) * 3);
}

function invalidateFrom(state, index) {
    for (let i = Math.max(0, index); i < state.clips.length; i++) {
        state.clips[i].validated = false;
    }
}

function currentResolutionFromWidgets(node) {
    const width = Number(getWidget(node, "width")?.value || 0);
    const height = Number(getWidget(node, "height")?.value || 0);
    if (!(width > 0) || !(height > 0)) return null;
    return effectiveManualResolution(width, height);
}

function invalidateForResolutionChange(node, runtime) {
    if (!node || !runtime?.state) return false;
    const expected = runtime.expectedResolution;
    const current = currentResolutionFromWidgets(node);
    if (!expected || !current) return false;

    const expectedW = Number(expected.width || 0);
    const expectedH = Number(expected.height || 0);
    if (!(expectedW > 0) || !(expectedH > 0)) return false;
    if (current.width === expectedW && current.height === expectedH) return false;

    const hadValidated = runtime.state.clips.some((clip) => Boolean(clip?.validated));
    const hadCached = Number(runtime.cachedCount || 0) > 0;

    // Once the requested geometry differs from the cache/project geometry,
    // every latent in that chain is incompatible. Reflect that immediately in
    // the cards instead of waiting for the backend to discover it at Queue.
    for (const clip of runtime.state.clips) clip.validated = false;
    runtime.validatedCount = 0;
    runtime.cachedCount = 0;
    runtime.resolutionInvalidated = true;
    runtime.statusText =
        `Resolution changed: ${expectedW}x${expectedH} → ${current.width}x${current.height} | ` +
        `clips invalidated; cache resets on next run`;

    if (hadValidated || hadCached) updateHidden(node, runtime);
    render(node, runtime);
    node.graph?.setDirtyCanvas(true, true);
    return hadValidated || hadCached;
}

function syncResolutionAndInvalidate(node, runtime) {
    syncResolutionMirror(node, runtime);
    invalidateForResolutionChange(node, runtime);
}


function advanceSeedAfterGenerate(clip) {
    const mode = String(clip?.seed_mode || "randomize");
    const max = Number.MAX_SAFE_INTEGER;
    const current = Math.max(0, Math.min(max, Math.trunc(Number(clip?.seed || 0))));

    if (mode === "randomize") {
        let next = randomSeed();
        // Extremely unlikely, but never leave the node cache-identical.
        if (next === current) next = (current + 1) % (max + 1);
        clip.seed = next;
    } else if (mode === "increment") {
        clip.seed = current >= max ? 0 : current + 1;
    } else if (mode === "decrement") {
        clip.seed = current <= 0 ? max : current - 1;
    }
    // fixed deliberately does nothing.
}

function cardStatus(runtime, clip, index) {
    if (
        Number(runtime.activeClipIndex) === index &&
        ["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""))
    ) {
        return "rendering";
    }

    const cached = index < Number(runtime.cachedCount || 0);
    if (clip.validated && cached) return "validated";
    const firstOpen = runtime.state.clips.findIndex((c) => !c.validated);
    if (index === firstOpen) return cached ? "candidate" : "current";
    if (cached) return "cached";
    return "future";
}

function updateHidden(node, runtime) {
    const raw = serializeState(runtime.state);
    runtime.jsonWidget.value = raw;
    node.graph?.setDirtyCanvas(true, true);
}

function updateRefsHidden(node, runtime) {
    if (!runtime?.refsWidget) return;
    runtime.refsState.refs = normalizeRefsArray(runtime.refsState?.refs || []);
    runtime.refsWidget.value = serializeRefsState(runtime.refsState);
    node.graph?.setDirtyCanvas(true, true);
}

function handleReferenceChange(node, runtime, message = "Image references changed") {
    if (!node || !runtime?.state) return;

    // Reference edits are deliberately user-controlled. Do not infer any
    // Ref-to-Clip relationship and do not change validation automatically.
    updateRefsHidden(node, runtime);

    // Auto resolution still follows the active guide ref. If the ref edit changes
    // the effective geometry, the existing resolution safety rule necessarily
    // invalidates the whole latent chain; that is independent of ref semantics.
    syncResolutionAndInvalidate(node, runtime);

    if (!runtime.resolutionInvalidated) {
        runtime.statusText = `${message} | validations unchanged`;
        render(node, runtime);
    }
}

function openReferenceEditor(node, runtime, slotIndex, ref) {
    if (!ref?.id || !node || !runtime) return;
    if (projectBusy(runtime) || runtime.refBusy || runtime.projectOperationBusy) {
        alert("Wait for the current clip generation to finish before editing a reference image.");
        return;
    }

    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "100000";
    overlay.style.background = "rgba(0,0,0,.86)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.padding = "24px";
    overlay.style.boxSizing = "border-box";

    const panel = document.createElement("div");
    panel.style.width = "min(1180px, 94vw)";
    panel.style.height = "min(820px, 92vh)";
    panel.style.minWidth = "0";
    panel.style.minHeight = "0";
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.background = "#191919";
    panel.style.border = "1px solid rgba(255,255,255,.18)";
    panel.style.borderRadius = "10px";
    panel.style.boxShadow = "0 18px 60px rgba(0,0,0,.65)";
    panel.style.overflow = "hidden";
    overlay.appendChild(panel);

    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "12px";
    header.style.padding = "10px 12px";
    header.style.borderBottom = "1px solid rgba(255,255,255,.12)";

    const title = document.createElement("div");
    title.textContent = `Reference Editor — Ref ${slotIndex + 1}`;
    title.style.fontWeight = "650";
    title.style.fontSize = "13px";
    title.style.overflow = "hidden";
    title.style.textOverflow = "ellipsis";
    title.style.whiteSpace = "nowrap";
    title.title = ref.original_name || `Ref ${slotIndex + 1}`;

    const closeButton = document.createElement("button");
    closeButton.textContent = "×";
    closeButton.title = "Close";
    closeButton.style.width = "28px";
    closeButton.style.minWidth = "28px";
    closeButton.style.height = "26px";
    closeButton.style.padding = "0";
    closeButton.style.fontSize = "18px";
    header.append(title, closeButton);
    panel.appendChild(header);

    const body = document.createElement("div");
    body.style.flex = "1 1 auto";
    body.style.minHeight = "0";
    body.style.minWidth = "0";
    body.style.display = "flex";
    body.style.gap = "0";
    panel.appendChild(body);

    const previewWrap = document.createElement("div");
    previewWrap.style.flex = "1 1 auto";
    previewWrap.style.minWidth = "0";
    previewWrap.style.minHeight = "0";
    previewWrap.style.display = "flex";
    previewWrap.style.alignItems = "center";
    previewWrap.style.justifyContent = "center";
    previewWrap.style.padding = "14px";
    previewWrap.style.boxSizing = "border-box";
    previewWrap.style.background = "#0f0f0f";

    const image = document.createElement("img");
    const sourceRef = { ...ref, id: ref.source_id || ref.id };
    image.src = refImageUrl(sourceRef);
    image.alt = ref.original_name || "Reference image";
    image.style.maxWidth = "100%";
    image.style.maxHeight = "100%";
    image.style.objectFit = "contain";
    image.style.borderRadius = "6px";
    image.style.boxShadow = "0 8px 30px rgba(0,0,0,.45)";
    image.draggable = false;
    previewWrap.appendChild(image);
    body.appendChild(previewWrap);

    const controls = document.createElement("div");
    controls.style.flex = "0 0 235px";
    controls.style.width = "235px";
    controls.style.boxSizing = "border-box";
    controls.style.padding = "14px";
    controls.style.borderLeft = "1px solid rgba(255,255,255,.12)";
    controls.style.display = "flex";
    controls.style.flexDirection = "column";
    controls.style.gap = "12px";
    controls.style.overflowY = "auto";
    body.appendChild(controls);

    const makeControl = (labelText) => {
        const wrap = document.createElement("div");
        wrap.style.display = "block";
        wrap.style.fontSize = "11px";
        wrap.style.fontWeight = "600";

        const headerRow = document.createElement("div");
        headerRow.style.display = "flex";
        headerRow.style.alignItems = "center";
        headerRow.style.justifyContent = "space-between";
        headerRow.style.gap = "8px";
        headerRow.style.marginBottom = "4px";

        const label = document.createElement("div");
        label.textContent = labelText;

        const number = document.createElement("input");
        number.type = "number";
        number.min = "0";
        number.max = "200";
        number.step = "1";
        number.value = "100";
        number.style.width = "58px";
        number.style.boxSizing = "border-box";
        number.style.padding = "3px 5px";
        number.style.borderRadius = "5px";
        number.style.border = "1px solid rgba(255,255,255,.18)";
        number.style.background = "rgba(0,0,0,.28)";
        number.style.color = "inherit";
        number.style.textAlign = "right";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "200";
        slider.step = "1";
        slider.value = "100";
        slider.style.width = "100%";
        slider.style.margin = "0";
        slider.style.padding = "0";
        slider.style.boxSizing = "border-box";
        slider.title = `${labelText}: 100`;

        headerRow.append(label, number);
        wrap.append(headerRow, slider);
        controls.appendChild(wrap);
        return { slider, number, labelText };
    };

    const saturation = makeControl("Saturation (%)");
    const contrast = makeControl("Contrast (%)");
    const brightness = makeControl("Brightness (%)");

    const help = document.createElement("div");
    help.textContent = "100 = original image. Edits are always calculated from the initially loaded reference, so Reset truly restores the original pixels.";
    help.style.fontSize = "10px";
    help.style.lineHeight = "1.35";
    help.style.opacity = ".66";
    controls.appendChild(help);

    const spacer = document.createElement("div");
    spacer.style.flex = "1 1 auto";
    controls.appendChild(spacer);

    const buttons = document.createElement("div");
    buttons.style.display = "grid";
    buttons.style.gridTemplateColumns = "1fr 1fr";
    buttons.style.gap = "7px";

    const reset = document.createElement("button");
    reset.textContent = "Reset";
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.style.gridColumn = "1 / -1";
    apply.style.fontWeight = "650";
    buttons.append(reset, cancel, apply);
    controls.appendChild(buttons);

    const numericValue = (control) => {
        const value = Number(control.slider.value);
        if (!Number.isFinite(value)) return 100;
        return Math.min(200, Math.max(0, value));
    };

    const setControlValue = (control, value) => {
        const parsed = Number(value);
        const clamped = Number.isFinite(parsed) ? Math.min(200, Math.max(0, parsed)) : 100;
        const text = String(Math.round(clamped));
        control.slider.value = text;
        control.number.value = text;
        control.slider.title = `${control.labelText}: ${text}`;
    };

    setControlValue(saturation, ref.saturation ?? 100);
    setControlValue(contrast, ref.contrast ?? 100);
    setControlValue(brightness, ref.brightness ?? 100);

    const updatePreview = () => {
        const b = numericValue(brightness);
        const c = numericValue(contrast);
        const sat = numericValue(saturation);
        image.style.filter = `brightness(${b}%) contrast(${c}%) saturate(${sat}%)`;
    };
    for (const control of [saturation, contrast, brightness]) {
        control.slider.addEventListener("input", () => {
            control.number.value = control.slider.value;
            control.slider.title = `${control.labelText}: ${control.slider.value}`;
            updatePreview();
        });
        control.number.addEventListener("input", () => {
            const value = Number(control.number.value);
            if (Number.isFinite(value)) {
                setControlValue(control, value);
                updatePreview();
            }
        });
        control.number.addEventListener("change", () => {
            setControlValue(control, control.number.value);
            updatePreview();
        });
    }
    updatePreview();

    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        window.removeEventListener("keydown", onKey);
        overlay.remove();
    };
    const onKey = (event) => {
        if (event.key === "Escape") close();
    };
    closeButton.addEventListener("click", close);
    cancel.addEventListener("click", close);
    overlay.addEventListener("click", close);
    panel.addEventListener("click", (event) => event.stopPropagation());
    window.addEventListener("keydown", onKey);

    reset.addEventListener("click", () => {
        setControlValue(saturation, 100);
        setControlValue(contrast, 100);
        setControlValue(brightness, 100);
        updatePreview();
    });

    apply.addEventListener("click", async () => {
        if (projectBusy(runtime) || runtime.refBusy || runtime.projectOperationBusy) {
            alert("Wait for the current clip generation to finish before editing a reference image.");
            return;
        }
        apply.disabled = true;
        reset.disabled = true;
        cancel.disabled = true;
        runtime.refBusy = true;
        runtime.statusText = `Applying Ref ${slotIndex + 1} adjustments…`;
        render(node, runtime);
        try {
            const response = await fetch(api.apiURL("/h3_extender/ref/edit"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ref_id: ref.id,
                    source_id: ref.source_id || ref.id,
                    original_name: ref.original_name || `ref_${slotIndex + 1}.png`,
                    saturation: numericValue(saturation),
                    contrast: numericValue(contrast),
                    brightness: numericValue(brightness),
                    external_signature: ref.external_signature || "",
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload?.ok || !payload?.ref) {
                throw new Error(payload?.error || `Reference edit failed (${response.status}).`);
            }
            const newRef = normalizeRefDescriptor(payload.ref);
            if (!newRef) throw new Error("The backend returned invalid reference metadata.");

            const current = runtime.refsState.refs[slotIndex];
            if (!current || String(current.id) !== String(ref.id)) {
                throw new Error(`Ref ${slotIndex + 1} changed while the editor was open.`);
            }

            runtime.refsState.refs[slotIndex] = newRef;
            if (sameRefContent(ref, newRef)) {
                updateRefsHidden(node, runtime);
                runtime.statusText = `Ref ${slotIndex + 1} unchanged`;
                render(node, runtime);
            } else {
                handleReferenceChange(node, runtime, `Ref ${slotIndex + 1} adjusted`);
            }
            close();
        } catch (error) {
            runtime.statusText = "Reference edit failed";
            render(node, runtime);
            alert(String(error?.message || error));
            apply.disabled = false;
            reset.disabled = false;
            cancel.disabled = false;
        } finally {
            runtime.refBusy = false;
            render(node, runtime);
        }
    });

    document.body.appendChild(overlay);
}

async function uploadReference(node, runtime, slotIndex, file) {
    if (!node || !runtime || !file) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before changing a reference image.");
        return;
    }

    runtime.refBusy = true;
    runtime.statusText = `Loading Ref ${slotIndex + 1}: ${file.name}…`;
    render(node, runtime);
    try {
        const form = new FormData();
        form.append("ref_file", file, file.name);
        const response = await fetch(api.apiURL("/h3_extender/ref/upload"), {
            method: "POST",
            body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok || !payload?.ref) {
            throw new Error(payload?.error || `Reference upload failed (${response.status}).`);
        }

        const newRef = normalizeRefDescriptor(payload.ref);
        if (!newRef) throw new Error("The backend returned invalid reference metadata.");
        const previous = runtime.refsState.refs[slotIndex];
        if (sameRefContent(previous, newRef)) {
            runtime.refsState.refs[slotIndex] = newRef;
            updateRefsHidden(node, runtime);
            runtime.statusText = `Ref ${slotIndex + 1} unchanged`;
            render(node, runtime);
            return;
        }

        runtime.refsState.refs[slotIndex] = newRef;
        handleReferenceChange(node, runtime, `Ref ${slotIndex + 1} loaded`);
    } catch (error) {
        runtime.statusText = "Reference load failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        runtime.refBusy = false;
        render(node, runtime);
    }
}

function removeReference(node, runtime, slotIndex) {
    if (!runtime?.refsState?.refs?.[slotIndex]) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before changing a reference image.");
        return;
    }
    const oldName = runtime.refsState.refs[slotIndex]?.original_name || `Ref ${slotIndex + 1}`;
    runtime.refsState.refs[slotIndex] = null;
    handleReferenceChange(node, runtime, `${oldName} removed`);
}

function nodeIs(node, className) {
    return node?.comfyClass === className || node?.type === className;
}

function connectedFinalDecode(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return null;
    const output = (node.outputs || []).find((o) => o?.name === "cache") || node.outputs?.[0];
    for (const linkId of output?.links || []) {
        const link = graph.links?.[linkId];
        if (!link) continue;
        const target = graph.getNodeById?.(link.target_id)
            || (graph._nodes || []).find((n) => String(n?.id) === String(link.target_id));
        if (target && nodeIs(target, FINAL_TARGET)) return target;
    }
    return null;
}

function colorMediaUrl(info) {
    const params = new URLSearchParams();
    params.set("filename", info?.filename || "");
    params.set("type", info?.type || "temp");
    params.set("subfolder", info?.subfolder || "");
    return api.apiURL("/view?" + params.toString());
}

function colorAtTimelineTime(timeline, time, targetIndex, liveAdjustment) {
    const t = Number(time || 0);
    for (const item of timeline || []) {
        const start = Number(item?.start || 0);
        const end = Number(item?.end || start);
        if (t >= start && t < end) {
            if (Number(item?.index) === Number(targetIndex)) return liveAdjustment;
            return normalizeColorAdjustment(item?.adjustment);
        }
    }
    return normalizeColorAdjustment();
}

function closeColorEditor(overlay) {
    try {
        const video = overlay?.querySelector?.("video");
        if (video) {
            video.pause();
            video.removeAttribute("src");
            video.load();
        }
    } catch (_) {}
    overlay?.remove?.();
}

async function openClipColorEditor(node, runtime, clipIndex) {
    const finalNode = connectedFinalDecode(node);
    if (!finalNode) {
        alert("Connect the Extender cache output to Final Decode / Preview first.");
        return;
    }

    const params = new URLSearchParams();
    params.set("owner_id", String(node.id));
    params.set("final_id", String(finalNode.id));
    params.set("clip_index", String(clipIndex));

    let payload;
    try {
        const response = await fetch(
            api.apiURL("/h3_extender/color_editor_info?" + params.toString())
        );
        payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Color editor failed (${response.status}).`);
        }
    } catch (error) {
        alert(`Color editor unavailable:\n${error?.message || error}`);
        return;
    }

    const timeline = Array.isArray(payload.timeline) ? payload.timeline : [];
    const target = timeline.find((item) => Number(item?.index) === Number(clipIndex));
    if (!target || !payload?.video?.filename) {
        alert("The decoded clip preview is not available yet.");
        return;
    }

    const clip = runtime.state.clips[clipIndex];
    let adjustment = normalizeColorAdjustment(
        clip?.color_adjustment || target?.adjustment
    );

    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "100000";
    overlay.style.background = "rgba(0,0,0,.78)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.padding = "24px";
    overlay.style.boxSizing = "border-box";

    const dialog = document.createElement("div");
    dialog.style.width = "min(1040px, 94vw)";
    dialog.style.maxHeight = "92vh";
    dialog.style.overflow = "auto";
    dialog.style.background = "#171717";
    dialog.style.color = "#f0f0f0";
    dialog.style.border = "1px solid rgba(255,255,255,.18)";
    dialog.style.borderRadius = "10px";
    dialog.style.boxShadow = "0 18px 60px rgba(0,0,0,.65)";
    dialog.style.padding = "14px";
    dialog.style.boxSizing = "border-box";

    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "12px";
    header.style.marginBottom = "10px";

    const title = document.createElement("strong");
    const clipName = String(clip?.name || "").trim();
    title.textContent = `Color Edit — Clip ${clipIndex + 1}${clipName ? ` — ${clipName}` : ""}`;
    title.style.fontSize = "15px";

    const close = document.createElement("button");
    close.textContent = "✕";
    close.title = "Close";
    close.style.width = "30px";
    close.style.height = "26px";
    close.style.cursor = "pointer";
    close.addEventListener("click", () => closeColorEditor(overlay));
    header.append(title, close);

    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "auto";
    video.style.display = "block";
    video.style.width = "100%";
    video.style.maxHeight = "58vh";
    video.style.objectFit = "contain";
    video.style.background = "#000";
    video.style.borderRadius = "6px";

    const totalEnd = timeline.length ? Number(timeline[timeline.length - 1]?.end || 0) : Number(target.end || 0);
    const loopStart = Math.max(0, Number(target.start || 0) - 2.0);
    const loopEnd = Math.min(totalEnd, Number(target.end || 0) + 2.0);

    const loopInfo = document.createElement("div");
    loopInfo.textContent = `Loop: ${loopStart.toFixed(2)}s → ${loopEnd.toFixed(2)}s  •  target ${Number(target.start).toFixed(2)}s → ${Number(target.end).toFixed(2)}s`;
    loopInfo.style.fontSize = "11px";
    loopInfo.style.opacity = ".72";
    loopInfo.style.margin = "7px 0 10px";

    const controls = document.createElement("div");
    controls.style.display = "grid";
    controls.style.gridTemplateColumns = "1fr";
    controls.style.gap = "8px";

    const valueInputs = {};
    const sliderRows = [];
    const makeSlider = (key, label, min, max) => {
        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "100px 1fr 64px";
        row.style.gap = "10px";
        row.style.alignItems = "center";

        const text = document.createElement("span");
        text.textContent = label;
        text.style.fontSize = "12px";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(min);
        slider.max = String(max);
        slider.step = "1";
        slider.value = String(Math.round(adjustment[key]));
        slider.style.width = "100%";

        const number = document.createElement("input");
        number.type = "number";
        number.min = String(min);
        number.max = String(max);
        number.step = "1";
        number.value = String(Math.round(adjustment[key]));
        number.style.width = "64px";
        number.style.boxSizing = "border-box";
        number.style.background = "rgba(0,0,0,.35)";
        number.style.color = "inherit";
        number.style.border = "1px solid rgba(255,255,255,.18)";
        number.style.borderRadius = "4px";
        number.style.padding = "3px 5px";

        const update = (raw) => {
            const n = Math.max(min, Math.min(max, Number(raw)));
            adjustment = { ...adjustment, [key]: Number.isFinite(n) ? n : 100 };
            slider.value = String(Math.round(adjustment[key]));
            number.value = String(Math.round(adjustment[key]));
            updateLiveFilter();
        };
        slider.addEventListener("input", () => update(slider.value));
        number.addEventListener("input", () => update(number.value));
        valueInputs[key] = { slider, number, update };
        row.append(text, slider, number);
        sliderRows.push(row);
        controls.appendChild(row);
    };

    const updateLiveFilter = () => {
        const c = colorAtTimelineTime(timeline, video.currentTime, clipIndex, adjustment);
        video.style.filter = cssColorFilter(c);
    };

    makeSlider("saturation", "Saturation", 0, 200);
    makeSlider("contrast", "Contrast", 50, 150);
    makeSlider("brightness", "Brightness", 50, 150);

    const buttons = document.createElement("div");
    buttons.style.display = "flex";
    buttons.style.justifyContent = "flex-end";
    buttons.style.gap = "8px";
    buttons.style.marginTop = "12px";

    const reset = document.createElement("button");
    reset.textContent = "Reset";
    reset.title = "Return this clip to neutral 100 / 100 / 100";
    reset.addEventListener("click", () => {
        adjustment = normalizeColorAdjustment();
        for (const [key, pair] of Object.entries(valueInputs)) {
            pair.slider.value = String(Math.round(adjustment[key]));
            pair.number.value = String(Math.round(adjustment[key]));
        }
        updateLiveFilter();
    });

    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => closeColorEditor(overlay));

    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.style.fontWeight = "700";
    apply.style.minWidth = "84px";
    apply.addEventListener("click", async () => {
        apply.disabled = true;
        apply.textContent = "Applying...";
        try {
            const response = await fetch(api.apiURL("/h3_extender/color_adjust"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    owner_id: String(node.id),
                    clip_index: Number(clipIndex),
                    adjustment: normalizeColorAdjustment(adjustment),
                }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok || !result?.ok) {
                throw new Error(result?.error || `Color adjustment failed (${response.status}).`);
            }
            clip.color_adjustment = normalizeColorAdjustment(result.adjustment);
            updateHidden(node, runtime);
            runtime.statusText = result.modified
                ? `Clip ${clipIndex + 1} color correction saved`
                : `Clip ${clipIndex + 1} color correction reset`;
            render(node, runtime);
            window.dispatchEvent(new CustomEvent("h3-extender-color-updated", {
                detail: {
                    owner_id: String(node.id),
                    color_timeline: Array.isArray(result.timeline) ? result.timeline : [],
                },
            }));
            node.graph?.setDirtyCanvas(true, true);
            closeColorEditor(overlay);
        } catch (error) {
            alert(`Color adjustment failed:\n${error?.message || error}`);
            apply.disabled = false;
            apply.textContent = "Apply";
        }
    });

    buttons.append(reset, cancel, apply);
    dialog.append(header, video, loopInfo, controls, buttons);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener("mousedown", (event) => {
        if (event.target === overlay) closeColorEditor(overlay);
    });
    const keyHandler = (event) => {
        if (event.key === "Escape" && overlay.isConnected) {
            closeColorEditor(overlay);
            document.removeEventListener("keydown", keyHandler);
        }
    };
    document.addEventListener("keydown", keyHandler);

    video.addEventListener("loadedmetadata", () => {
        video.currentTime = loopStart;
        updateLiveFilter();
        video.play().catch(() => {});
    });
    video.addEventListener("timeupdate", () => {
        if (video.currentTime >= loopEnd - 0.015 || video.currentTime < loopStart - 0.05) {
            video.currentTime = loopStart;
        }
        updateLiveFilter();
    });
    video.addEventListener("seeked", updateLiveFilter);
    if (typeof video.requestVideoFrameCallback === "function") {
        const colorFrameTick = () => {
            if (!overlay.isConnected) return;
            if (video.currentTime >= loopEnd - 0.015 || video.currentTime < loopStart - 0.05) {
                video.currentTime = loopStart;
            }
            updateLiveFilter();
            video.requestVideoFrameCallback(colorFrameTick);
        };
        video.requestVideoFrameCallback(colorFrameTick);
    }
    video.src = colorMediaUrl(payload.video) + "&t=" + Date.now();
    video.load();
}

function collectWidgetValues(node, names) {
    const out = {};
    for (const name of names) {
        const widget = getWidget(node, name);
        if (widget) out[name] = widget.value;
    }
    return out;
}

function collectConnectionSummary(node) {
    const out = {};
    for (const input of node?.inputs || []) {
        out[String(input?.name || "")] = input?.link != null;
    }
    return out;
}

function collectProjectPayload(node, runtime) {
    updateHidden(node, runtime);
    updateRefsHidden(node, runtime);
    const finalNode = connectedFinalDecode(node);
    const settings = collectWidgetValues(node, PROJECT_WIDGETS);
    // In Auto mode the visible width/height widgets are mirrors of the active
    // derived resolution. Preserve the user's Manual fallback separately so a
    // later Auto -> Manual switch restores what they actually entered.
    settings.width = Number(runtime.manualWidth || settings.width || 896);
    settings.height = Number(runtime.manualHeight || settings.height || 576);
    return {
        schema_version: 2,
        extender: {
            class_name: TARGET,
            node_title: String(node?.title || "MiniMax H3 Extender"),
            settings,
            resolution: {
                mode: String(getWidget(node, "resolution_mode")?.value || "manual"),
                megapixels: Number(getWidget(node, "megapixels")?.value ?? 0.40),
                manual_width: Number(runtime.manualWidth || settings.width || 0),
                manual_height: Number(runtime.manualHeight || settings.height || 0),
                resolved_width: Number(runtime.resolvedWidth || runtime.expectedResolution?.width || 0),
                resolved_height: Number(runtime.resolvedHeight || runtime.expectedResolution?.height || 0),
                guide_ref: String(runtime.resolutionGuide || ""),
                fallback: Boolean(runtime.resolutionFallback),
            },
            clips_json: serializeState(runtime.state),
            clips: runtime.state.clips.map((clip) => ({ ...clip })),
            refs_json: serializeRefsState(runtime.refsState),
            references: runtime.refsState.refs.map((ref) => ref ? { ...ref } : null),
            connections: collectConnectionSummary(node),
        },
        final_decode: finalNode ? {
            class_name: FINAL_TARGET,
            settings: collectWidgetValues(finalNode, FINAL_PROJECT_WIDGETS),
        } : null,
    };
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget || value === undefined) return false;
    widget.value = value;
    return true;
}

function applyProjectPayload(node, runtime, projectPayload) {
    const extender = projectPayload?.extender || {};
    const settings = extender?.settings || {};
    if (typeof extender?.node_title === "string" && extender.node_title.trim()) {
        node.title = extender.node_title;
    }
    for (const name of PROJECT_WIDGETS) {
        if (name === "clips_json" || name === "refs_json") continue;
        if (Object.prototype.hasOwnProperty.call(settings, name)) {
            setWidgetValue(node, name, settings[name]);
        }
    }

    // v14.24 and older .ext projects did not know about automatic resolution.
    // Preserve their exact behavior instead of silently deriving a new size.
    if (!Object.prototype.hasOwnProperty.call(settings, "resolution_mode")) {
        setWidgetValue(node, "resolution_mode", "manual");
    }

    const savedResolution = extender?.resolution;
    const savedManualW = Number(savedResolution?.manual_width || settings?.width || 0);
    const savedManualH = Number(savedResolution?.manual_height || settings?.height || 0);
    rememberManualResolution(node, runtime, savedManualW, savedManualH);
    const savedW = Number(savedResolution?.resolved_width || 0);
    const savedH = Number(savedResolution?.resolved_height || 0);
    if (savedW > 0 && savedH > 0) {
        runtime.expectedResolution = { width: savedW, height: savedH };
        // Loading a portable project is the one place where the archived
        // geometry is authoritative. Put that exact size in Manual mode so the
        // imported latent cache can continue unchanged. The user can switch
        // back to Auto or edit width/height afterwards; doing so starts a new
        // cache at the newly requested resolution.
        setWidgetValue(node, "width", savedW);
        setWidgetValue(node, "height", savedH);
        setWidgetValue(node, "resolution_mode", "manual");
        rememberManualResolution(node, runtime, savedW, savedH);
        runtime.projectResolutionLoaded = true;
    }

    const rawRefs =
        extender?.refs_json
        || settings?.refs_json
        || JSON.stringify({ version: 2, refs: extender?.references || [] });
    runtime.refsState = parseRefsState(rawRefs);
    updateRefsHidden(node, runtime);

    const rawClips = String(
        extender?.clips_json
        || settings?.clips_json
        || JSON.stringify({ version: 1, clips: extender?.clips || [] })
    );
    runtime.state = parseState(rawClips);
    // Loading a project mutates the disk cache outside ComfyUI's executor. A
    // one-shot token forces the Extender input hash to change even if every
    // visible setting happens to match the workflow that was previously run.
    runtime.state.load_token = `${Date.now().toString(36)}_${randomSeed().toString(36)}`;
    updateHidden(node, runtime);

    const finalSettings = projectPayload?.final_decode?.settings;
    const finalNode = connectedFinalDecode(node);
    if (finalNode && finalSettings && typeof finalSettings === "object") {
        for (const name of FINAL_PROJECT_WIDGETS) {
            if (Object.prototype.hasOwnProperty.call(finalSettings, name)) {
                setWidgetValue(finalNode, name, finalSettings[name]);
            }
        }
        finalNode.graph?.setDirtyCanvas(true, true);
    }

    node.graph?.setDirtyCanvas(true, true);
}

function projectBusy(runtime) {
    return ["preparing", "sampling", "complete"].includes(String(runtime?.activePhase || ""));
}

function setProjectButtonsBusy(runtime, busy) {
    if (!runtime) return;
    runtime.projectOperationBusy = Boolean(busy);
    if (runtime.saveProjectButton) runtime.saveProjectButton.disabled = Boolean(busy);
    if (runtime.loadProjectButton) runtime.loadProjectButton.disabled = Boolean(busy);
}

async function saveProject(node, runtime) {
    if (!node || !runtime) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before saving the project.");
        return;
    }
    if (runtime.resolutionInvalidated) {
        alert(
            "The resolution has changed and the previous cache is no longer compatible. " +
            "Queue the Extender once to start the new-resolution cache before saving the project."
        );
        return;
    }

    const suggested = runtime.projectName || "MiniMax_H3_Project";
    const requested = prompt("Project name (.ext)", suggested);
    if (requested == null) return;
    const projectName = String(requested || suggested).trim() || suggested;

    setProjectButtonsBusy(runtime, true);
    runtime.statusText = "Saving project…";
    render(node, runtime);
    try {
        const response = await fetch(api.apiURL("/h3_extender/project/prepare_save"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                owner_id: String(node.id),
                project_name: projectName,
                project: collectProjectPayload(node, runtime),
            }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok || !payload?.token) {
            throw new Error(payload?.error || `Save Project failed (${response.status}).`);
        }

        runtime.projectName = String(payload.filename || projectName).replace(/\.ext$/i, "");
        node.properties = node.properties || {};
        node.properties.h3_project_name = runtime.projectName;
        runtime.statusText = `Project ready: ${payload.filename || projectName} | refs ${Number(payload?.references?.count ?? refCount(runtime))} embedded`;
        render(node, runtime);

        // Do not fetch the archive into a JS Blob: .ext files may be many GB.
        // A normal browser download streams it directly from the backend.
        const a = document.createElement("a");
        a.href = api.apiURL(
            "/h3_extender/project/download?token=" + encodeURIComponent(String(payload.token))
        );
        a.download = String(payload.filename || `${runtime.projectName}.ext`);
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        setTimeout(() => a.remove(), 0);
    } catch (error) {
        runtime.statusText = "Save Project failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        setProjectButtonsBusy(runtime, false);
        render(node, runtime);
    }
}

async function loadProjectFile(node, runtime, file) {
    if (!node || !runtime || !file) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before loading a project.");
        return;
    }
    if (!confirm(
        "Load this .ext project?\n\nThe current Extender cache, image references and project settings will be replaced."
    )) return;

    setProjectButtonsBusy(runtime, true);
    runtime.statusText = `Loading ${file.name}…`;
    render(node, runtime);
    try {
        const form = new FormData();
        form.append("owner_id", String(node.id));
        form.append("project_file", file, file.name);
        const response = await fetch(api.apiURL("/h3_extender/project/load"), {
            method: "POST",
            body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Load Project failed (${response.status}).`);
        }

        applyProjectPayload(node, runtime, payload.project || {});
        runtime.cachedCount = Number(payload?.cache?.cached_count || 0);
        runtime.validatedCount = Number(payload?.cache?.validated_count || 0);
        const loadedW = Number(payload?.cache?.resolved_width || runtime.expectedResolution?.width || 0);
        const loadedH = Number(payload?.cache?.resolved_height || runtime.expectedResolution?.height || 0);
        if (loadedW > 0 && loadedH > 0) {
            runtime.expectedResolution = { width: loadedW, height: loadedH };
            setWidgetValue(node, "width", loadedW);
            setWidgetValue(node, "height", loadedH);
            setWidgetValue(node, "resolution_mode", "manual");
            rememberManualResolution(node, runtime, loadedW, loadedH);
            runtime.resolutionMirrorActive = false;
            runtime.projectResolutionLoaded = true;
            runtime.resolutionInvalidated = false;
        }
        runtime.cacheStateRestored = true;
        runtime.projectName = String(payload.project_name || file.name).replace(/\.ext$/i, "");
        node.properties = node.properties || {};
        node.properties.h3_project_name = runtime.projectName;
        const resolutionText = loadedW > 0 && loadedH > 0 ? ` | ${loadedW}x${loadedH}` : "";
        runtime.statusText =
            `Loaded ${runtime.projectName}${resolutionText} | refs ${refCount(runtime)} | cached ${runtime.cachedCount}/${runtime.state.clips.length} | ` +
            `validated ${runtime.validatedCount}`;
        render(node, runtime);
        syncDomHeight(node, runtime, false);

        // Final Decode / Preview can rebuild the full preview from decoded blobs
        // already inside the imported cache, with no sampler or VAE execution.
        window.dispatchEvent(new CustomEvent("h3-extender-project-loaded", {
            detail: { owner_id: String(node.id) },
        }));
    } catch (error) {
        runtime.statusText = "Load Project failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        setProjectButtonsBusy(runtime, false);
        render(node, runtime);
    }
}

function makeFieldLabel(text) {
    const label = document.createElement("div");
    label.textContent = text;
    label.style.fontSize = "11px";
    label.style.opacity = "0.72";
    label.style.margin = "5px 0 3px";
    return label;
}

function makeNumberInput(value, min, max, step) {
    const input = document.createElement("input");
    input.type = "number";
    input.value = String(value);
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.background = "rgba(0,0,0,.25)";
    input.style.border = "1px solid rgba(255,255,255,.15)";
    input.style.color = "inherit";
    input.style.borderRadius = "5px";
    input.style.padding = "5px 7px";
    return input;
}

function renderReferences(node, runtime) {
    const row = runtime?.refsRow;
    if (!row) return;
    row.replaceChildren();

    const refs = runtime.refsState?.refs || [];

    for (let index = 0; index < MAX_IMAGE_REFS; index++) {
        const ref = refs[index] || null;
        const slot = document.createElement("div");
        // Fill the whole available node width with nine equal reference slots.
        // REF_SLOT_WIDTH is a hard minimum for each slot, not for the node.
        // The node itself may shrink well below the combined strip width; once
        // that happens this row owns the horizontal overflow and exposes its scrollbar.
        slot.style.flex = "1 1 0px";
        slot.style.minWidth = `${REF_SLOT_WIDTH}px`;
        slot.style.boxSizing = "border-box";
        slot.style.position = "relative";

        const load = document.createElement("button");
        load.textContent = ref ? `Replace Ref ${index + 1}` : `Load Ref ${index + 1}`;
        load.title = ref
            ? `Replace Ref ${index + 1}: ${ref.original_name || "reference"}`
            : `Load image reference ${index + 1}`;
        load.style.width = "100%";
        load.style.height = "23px";
        load.style.padding = "2px 4px";
        load.style.fontSize = "10px";
        load.disabled = Boolean(
            runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime)
        );
        load.addEventListener("click", (event) => {
            event.preventDefault();
            if (load.disabled) return;
            runtime.pendingRefSlot = index;
            runtime.refFileInput?.click();
        });
        slot.appendChild(load);

        const thumb = document.createElement("div");
        thumb.style.marginTop = "2px";
        thumb.style.width = "100%";
        thumb.style.height = `${REF_THUMB_HEIGHT}px`;
        thumb.style.boxSizing = "border-box";
        thumb.style.border = "1px solid rgba(255,255,255,.15)";
        thumb.style.borderRadius = "6px";
        thumb.style.background = "rgba(0,0,0,.24)";
        thumb.style.display = "flex";
        thumb.style.alignItems = "center";
        thumb.style.justifyContent = "center";
        thumb.style.position = "relative";
        thumb.style.overflow = "hidden";

        if (ref) {
            const img = document.createElement("img");
            img.src = refImageUrl(ref);
            img.alt = ref.original_name || `Ref ${index + 1}`;
            img.title = `${ref.original_name || `Ref ${index + 1}`} — double-click to edit`;
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
            img.style.cursor = "pointer";
            img.draggable = false;
            img.addEventListener("dblclick", (event) => {
                event.preventDefault();
                event.stopPropagation();
                openReferenceEditor(node, runtime, index, ref);
            });
            thumb.appendChild(img);

            const remove = document.createElement("button");
            remove.textContent = "×";
            remove.title = `Remove Ref ${index + 1}`;
            remove.style.position = "absolute";
            remove.style.top = "3px";
            remove.style.right = "3px";
            remove.style.width = "20px";
            remove.style.height = "20px";
            remove.style.minWidth = "20px";
            remove.style.padding = "0";
            remove.style.lineHeight = "16px";
            remove.style.borderRadius = "10px";
            remove.style.background = "rgba(0,0,0,.68)";
            remove.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
            remove.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                removeReference(node, runtime, index);
            });
            thumb.appendChild(remove);
        } else {
            const empty = document.createElement("span");
            empty.textContent = "+";
            empty.style.fontSize = "24px";
            empty.style.opacity = ".55";
            thumb.appendChild(empty);
        }
        slot.appendChild(thumb);

        const meta = document.createElement("div");
        meta.style.marginTop = "1px";
        meta.style.fontSize = "9px";
        meta.style.lineHeight = "9px";
        meta.style.opacity = ".6";
        meta.style.textAlign = "center";
        meta.style.whiteSpace = "nowrap";
        meta.style.overflow = "hidden";
        meta.style.textOverflow = "ellipsis";
        meta.textContent = ref && ref.width > 0 && ref.height > 0
            ? `${Math.trunc(ref.width)}×${Math.trunc(ref.height)}`
            : "empty";
        meta.title = ref?.original_name || "";
        slot.appendChild(meta);

        row.appendChild(slot);
    }
}

function render(node, runtime) {
    const { state, cards, counter, status } = runtime;
    renderReferences(node, runtime);
    cards.replaceChildren();

    counter.textContent = `${state.clips.length} clip${state.clips.length > 1 ? "s" : ""} • ${refCount(runtime)} ref${refCount(runtime) === 1 ? "" : "s"}`;
    status.textContent = runtime.statusText || "Ready";

    state.clips.forEach((clip, index) => {
        const card = document.createElement("div");
        card.className = "h3-extender-card";
        card.dataset.clipIndex = String(index);
        card.style.flex = `0 0 ${CARD_WIDTH}px`;
        card.style.width = `${CARD_WIDTH}px`;
        card.style.boxSizing = "border-box";
        card.style.padding = "9px";
        card.style.borderRadius = "8px";
        card.style.background = "rgba(20,20,20,.72)";
        card.style.border = "1px solid rgba(255,255,255,.13)";
        card.style.display = "flex";
        card.style.flexDirection = "column";
        card.style.minHeight = `${CARD_MIN_HEIGHT}px`;

        const st = cardStatus(runtime, clip, index);
        if (st === "rendering") {
            card.style.border = "3px solid rgba(70,210,255,1)";
            card.style.boxShadow = "0 0 0 1px rgba(70,210,255,.25), 0 0 16px rgba(70,210,255,.38)";
            card.style.background = "rgba(24,40,46,.88)";
        } else if (st === "validated") {
            card.style.borderColor = "rgba(80,210,120,.8)";
        } else if (st === "candidate" || st === "current") {
            card.style.borderColor = "rgba(255,180,60,.9)";
        } else if (st === "cached") {
            card.style.borderColor = "rgba(90,155,230,.65)";
        }

        const head = document.createElement("div");
        head.style.display = "flex";
        head.style.alignItems = "center";
        head.style.gap = "7px";
        head.style.marginBottom = "5px";

        const title = document.createElement("strong");
        title.textContent = `CLIP ${index + 1}`;
        title.style.flex = "0 0 auto";
        title.style.whiteSpace = "nowrap";

        const name = document.createElement("input");
        name.type = "text";
        name.value = clip.name || "";
        name.placeholder = "name";
        name.title = "Optional clip/card name";
        name.style.flex = "1 1 0";
        name.style.minWidth = "0";
        name.style.height = "22px";
        name.style.boxSizing = "border-box";
        name.style.background = "rgba(0,0,0,.22)";
        name.style.border = "1px solid rgba(255,255,255,.12)";
        name.style.color = "inherit";
        name.style.borderRadius = "4px";
        name.style.padding = "2px 5px";
        name.style.fontSize = "11px";
        name.addEventListener("input", () => {
            if (name.value === clip.name) return;
            clip.name = name.value;
            updateHidden(node, runtime);
            // Keep focus while typing; no DOM rebuild here.
        });

        const colorWrap = document.createElement("div");
        colorWrap.style.display = "flex";
        colorWrap.style.alignItems = "center";
        colorWrap.style.gap = "2px";
        colorWrap.style.flex = "0 0 auto";

        const colorButton = document.createElement("button");
        colorButton.type = "button";
        colorButton.textContent = "🎨";
        const colorBusy = ["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""));
        const colorCached = index < Number(runtime.cachedCount || 0);
        colorButton.title = colorBusy
            ? "Color editing is disabled while the Extender is rendering"
            : colorCached
                ? "Edit color for this decoded clip"
                : "Color editor becomes available after this clip has been decoded";
        colorButton.disabled = colorBusy || !colorCached;
        colorButton.style.width = "27px";
        colorButton.style.height = "22px";
        colorButton.style.padding = "0";
        colorButton.style.borderRadius = "4px";
        colorButton.style.cursor = colorButton.disabled ? "default" : "pointer";
        colorButton.style.opacity = colorButton.disabled ? ".35" : ".9";
        colorButton.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (!colorButton.disabled) openClipColorEditor(node, runtime, index);
        });

        const colorCheck = document.createElement("span");
        colorCheck.textContent = colorAdjustmentIsNeutral(clip.color_adjustment) ? "" : "✓";
        colorCheck.title = colorCheck.textContent ? "Color correction applied" : "";
        colorCheck.style.width = "10px";
        colorCheck.style.fontSize = "11px";
        colorCheck.style.fontWeight = "700";
        colorCheck.style.color = "rgba(115,225,145,.95)";
        colorCheck.style.textAlign = "center";

        colorWrap.append(colorButton, colorCheck);

        const badge = document.createElement("span");
        badge.style.fontSize = "10px";
        badge.style.opacity = ".8";
        badge.textContent =
            st === "rendering"
                ? (
                    runtime.activePhase === "preparing"
                        ? "◆ PREPARING"
                        : runtime.activePhase === "complete"
                            ? "✓ COMPLETE"
                            : "▶ RENDERING"
                ) :
            st === "validated" ? "VALIDATED" :
            st === "candidate" ? "● CANDIDATE" :
            st === "current" ? "● NEXT" :
            st === "cached" ? "CACHE" : "○";

        head.append(title, name, colorWrap, badge);
        card.appendChild(head);

        card.appendChild(makeFieldLabel("Prompt"));
        const prompt = document.createElement("textarea");
        prompt.value = clip.prompt;
        prompt.spellcheck = false;
        prompt.style.width = "100%";
        // The prompt area is the flexible part of each card. When the node is
        // stretched vertically, extra height goes here instead of becoming an
        // empty block below the card controls.
        prompt.style.height = "190px";
        prompt.style.minHeight = "190px";
        prompt.style.flex = "1 1 190px";
        prompt.style.resize = "vertical";
        prompt.style.boxSizing = "border-box";
        prompt.style.background = "rgba(0,0,0,.27)";
        prompt.style.border = "1px solid rgba(255,255,255,.15)";
        prompt.style.color = "inherit";
        prompt.style.borderRadius = "5px";
        prompt.style.padding = "6px";
        prompt.addEventListener("input", () => {
            if (prompt.value === clip.prompt) return;
            clip.prompt = prompt.value;
            updateHidden(node, runtime);
            // Do not rebuild the DOM while typing: that would steal focus.
        });
        prompt.addEventListener("blur", () => render(node, runtime));
        card.appendChild(prompt);

        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "1fr 92px";
        row.style.gap = "7px";
        row.style.alignItems = "end";

        const seedBox = document.createElement("div");
        seedBox.appendChild(makeFieldLabel("Seed"));
        const seedRow = document.createElement("div");
        seedRow.style.display = "flex";
        seedRow.style.gap = "5px";
        const seed = makeNumberInput(clip.seed, 0, Number.MAX_SAFE_INTEGER, 1);
        seed.style.minWidth = "0";
        seed.addEventListener("change", () => {
            const v = Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(Number(seed.value || 0))));
            if (v !== clip.seed) {
                clip.seed = v;
                updateHidden(node, runtime);
                render(node, runtime);
            }
        });
        const dice = document.createElement("button");
        dice.textContent = "🎲";
        dice.title = "Randomize seed";
        dice.style.width = "32px";
        dice.addEventListener("click", (e) => {
            e.preventDefault();
            clip.seed = randomSeed();
            updateHidden(node, runtime);
            render(node, runtime);
        });
        seedRow.append(seed, dice);
        seedBox.appendChild(seedRow);

        const seedMode = document.createElement("select");
        seedMode.title = "Seed behavior after a generated candidate";
        seedMode.style.width = "100%";
        seedMode.style.marginTop = "4px";
        seedMode.style.boxSizing = "border-box";
        seedMode.style.background = "rgba(0,0,0,.25)";
        seedMode.style.border = "1px solid rgba(255,255,255,.15)";
        seedMode.style.color = "inherit";
        seedMode.style.borderRadius = "5px";
        seedMode.style.padding = "4px 5px";
        for (const [value, label] of [
            ["randomize", "after: randomize"],
            ["fixed", "after: fixed"],
            ["increment", "after: increment"],
            ["decrement", "after: decrement"],
        ]) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            seedMode.appendChild(option);
        }
        seedMode.value = clip.seed_mode || "randomize";
        seedMode.addEventListener("change", () => {
            clip.seed_mode = seedMode.value;
            updateHidden(node, runtime);
        });
        seedBox.appendChild(seedMode);

        const durBox = document.createElement("div");
        durBox.appendChild(makeFieldLabel("Duration s"));
        const duration = makeNumberInput(clip.duration, 0.25, 150, 0.1);
        duration.addEventListener("change", () => {
            const v = Math.max(0.25, Math.min(150, Number(duration.value || 10)));
            if (Math.abs(v - clip.duration) > 1e-9) {
                clip.duration = v;
                updateHidden(node, runtime);
                render(node, runtime);
            }
        });
        durBox.appendChild(duration);
        row.append(seedBox, durBox);
        card.appendChild(row);

        const foot = document.createElement("div");
        foot.style.display = "flex";
        foot.style.alignItems = "center";
        foot.style.justifyContent = "space-between";
        foot.style.marginTop = "9px";

        const validateLabel = document.createElement("label");
        validateLabel.style.display = "flex";
        validateLabel.style.alignItems = "center";
        validateLabel.style.gap = "6px";
        validateLabel.style.cursor = "pointer";
        const validated = document.createElement("input");
        validated.type = "checkbox";
        validated.checked = clip.validated;
        validated.addEventListener("change", () => {
            if (validated.checked) {
                clip.validated = true;
            } else {
                invalidateFrom(state, index);
            }
            // A valid chain is necessarily a continuous validated prefix.
            let open = false;
            for (const c of state.clips) {
                if (open) c.validated = false;
                else if (!c.validated) open = true;
            }
            updateHidden(node, runtime);
            render(node, runtime);
        });
        validateLabel.append(validated, document.createTextNode("Validated"));

        const info = document.createElement("span");
        const rawFrames = Math.max(5, Math.round(clip.duration * 24));
        let aligned = rawFrames;
        while (aligned % 17 !== 5) aligned++;
        info.textContent = `${aligned}f / ${(aligned / 24).toFixed(3)}s`;
        info.style.fontSize = "10px";
        info.style.opacity = ".65";

        foot.append(validateLabel, info);
        card.appendChild(foot);
        cards.appendChild(card);
    });
}

function syncDomHeight(node, runtime, forceMin = false, retry = 0) {
    if (!node || !runtime?.domWidget || runtime.syncingDomHeight) return;

    const mode = domWidgetRenderMode(runtime.root);
    if (mode === "pending") {
        if (retry < 12) {
            requestAnimationFrame(() => syncDomHeight(node, runtime, forceMin, retry + 1));
        }
        return;
    }

    // Nodes 2.0 owns the DOM-widget row height. Never derive a new getHeight
    // value from node.size here: node.size -> DOM getHeight -> node.size is the
    // feedback loop that created the infinite-height nodes.
    if (mode === "nodes2") {
        const currentH = Number(node.size?.[1] || 0);
        const y = Number(runtime.domWidget.last_y);
        const fallbackH = Number.isFinite(y) && y > 0
            ? y + NODES2_MIN_HEIGHT + BOTTOM_PAD
            : NODES2_MIN_HEIGHT + 180;

        // One-time recovery for workflows that were already saved with a
        // runaway height by an older build. This is not DOM-driven resizing;
        // it only removes a clearly corrupted value.
        if (
            runtime.lastRenderMode !== "nodes2" &&
            obviouslyPoisonedHeight(currentH, fallbackH)
        ) {
            runtime.syncingDomHeight = true;
            try {
                const rememberedLegacyH = Number(runtime.legacyNodeHeight);
                const targetH = (
                    Number.isFinite(rememberedLegacyH) &&
                    !obviouslyPoisonedHeight(rememberedLegacyH, fallbackH)
                )
                    ? Math.max(fallbackH, rememberedLegacyH)
                    : fallbackH;
                const targetW = Math.max(
                    NODE_MIN_WIDTH,
                    Number(node.size?.[0] || NODE_MIN_WIDTH)
                );
                node.setSize([targetW, targetH]);
            } finally {
                runtime.syncingDomHeight = false;
            }
        }

        runtime.lastRenderMode = "nodes2";

        // Nodes 2.0 mounts this element inside WidgetDOM.vue's flex wrapper
        // (`flex flex-col *:flex-1`) and NodeWidgets.vue owns the grid row.
        // Do NOT use percentage heights here. A `height: 100%` has no stable
        // intrinsic size while CSS Grid is resolving an `auto` row; after a
        // manual resize that row can collapse to 0 and WidgetDOM will not
        // remount the element until a page refresh. Keep a real intrinsic
        // minimum instead and let Vue stretch the row/child naturally.
        runtime.root.style.height = "auto";
        runtime.root.style.minHeight = `${NODES2_MIN_HEIGHT}px`;
        runtime.root.style.setProperty("--comfy-widget-min-height", `${NODES2_MIN_HEIGHT}px`);
        runtime.root.style.maxHeight = "none";
        runtime.root.style.flex = "1 1 auto";
        runtime.root.style.paddingTop = `${5 + NODES2_TOP_GAP}px`;
        // Avoid a second vertical clipping boundary at fractional canvas zooms.
        // Horizontal clipping/scrolling is still owned by `cards`.
        runtime.root.style.overflow = "visible";

        runtime.cards.style.height = "auto";
        runtime.cards.style.flex = "1 1 auto";
        // The horizontal scrollbar has reserved space below the cards. Give the
        // row enough intrinsic height for both the card and that gutter so the
        // top/bottom cannot be shaved off by grid rounding at certain zooms.
        runtime.cards.style.minHeight = `${NODES2_CARDS_MIN_HEIGHT}px`;
        return;
    }

    const y = Number(runtime.domWidget.last_y);
    if (!Number.isFinite(y) || y <= 0) {
        if (retry < 12) {
            requestAnimationFrame(() => syncDomHeight(node, runtime, forceMin, retry + 1));
        }
        return;
    }

    // Remove Nodes 2.0-only intrinsic sizing when returning to Legacy.
    runtime.root.style.paddingTop = "5px";
    runtime.root.style.minHeight = "0";
    runtime.root.style.setProperty("--comfy-widget-min-height", `${UI_MIN_HEIGHT}px`);
    runtime.root.style.maxHeight = "none";
    runtime.root.style.flex = "0 0 auto";
    runtime.root.style.overflow = "hidden";

    runtime.syncingDomHeight = true;
    try {
        let w = Math.max(NODE_MIN_WIDTH, Number(node.size?.[0] || NODE_MIN_WIDTH));
        let h = Number(node.size?.[1] || 0);
        const minNodeH = y + UI_MIN_HEIGHT + BOTTOM_PAD;
        const returningFromNodes2 = runtime.lastRenderMode === "nodes2";

        if (returningFromNodes2) {
            // Restore the last real Legacy height. If this node was first opened
            // in Nodes 2.0 (so there is no stored Legacy size), start from the
            // calculated Legacy minimum instead of inheriting a Vue runaway.
            const rememberedLegacyH = Number(runtime.legacyNodeHeight);
            h = (
                Number.isFinite(rememberedLegacyH) &&
                !obviouslyPoisonedHeight(rememberedLegacyH, minNodeH)
            )
                ? Math.max(minNodeH, rememberedLegacyH)
                : minNodeH;
        } else if (
            runtime.lastRenderMode == null &&
            obviouslyPoisonedHeight(h, minNodeH)
        ) {
            // Also heal workflows that are opened directly in Legacy after an
            // older version serialized an absurd height.
            h = minNodeH;
        } else if (forceMin && h < minNodeH) {
            h = minNodeH;
        }

        if (w !== Number(node.size?.[0]) || h !== Number(node.size?.[1])) {
            node.setSize([w, h]);
        }

        const actualH = Number(node.size?.[1] || h);
        const available = Math.max(UI_MIN_HEIGHT, actualH - y - BOTTOM_PAD);
        runtime.root.style.height = `${available}px`;
        runtime.cards.style.height = `${Math.max(340, available - 55 - REF_SECTION_HEIGHT)}px`;
        runtime.cards.style.flex = "0 0 auto";
        runtime.cards.style.minHeight = "";
        runtime.domHeight = available;
        if (!obviouslyPoisonedHeight(actualH, minNodeH)) {
            runtime.legacyNodeHeight = actualH;
        }
        runtime.lastRenderMode = "legacy";
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        runtime.syncingDomHeight = false;
    }
}

function installInvalidationHooks(node, runtime) {
    // Image references are no longer graph sockets. Other input/parameter
    // changes deliberately preserve explicit clip validation as before.
}


function buildUi(node) {
    if (node.__h3Extender) return node.__h3Extender;

    const jsonWidget = getWidget(node, "clips_json");
    const refsWidget = getWidget(node, "refs_json");
    if (!jsonWidget || !refsWidget) return null;
    hideNativeWidget(node, jsonWidget);
    hideNativeWidget(node, refsWidget);

    const state = parseState(jsonWidget.value);
    const refsState = parseRefsState(refsWidget.value);

    const root = document.createElement("div");
    root.style.width = "100%";
    root.style.minWidth = "0";
    root.style.height = `${UI_MIN_HEIGHT}px`;
    root.style.minHeight = `${UI_MIN_HEIGHT}px`;
    // Official DOMWidgetImpl.computeLayoutSize() reads this CSS variable as a
    // fallback to getMinHeight. Keeping both makes the intrinsic contract clear
    // to current and slightly older Nodes 2.0 frontends.
    root.style.setProperty("--comfy-widget-min-height", `${NODES2_MIN_HEIGHT}px`);
    root.style.boxSizing = "border-box";
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.padding = "5px 0 0";
    root.style.overflow = "hidden";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.minWidth = "0";
    toolbar.style.gap = "7px";
    toolbar.style.alignItems = "center";
    toolbar.style.marginBottom = "7px";

    const add = document.createElement("button");
    add.textContent = "+ Add Clip";
    add.addEventListener("click", (e) => {
        e.preventDefault();
        runtime.state.clips.push(newClip(runtime.state.clips.length));
        updateHidden(node, runtime);
        render(node, runtime);
        requestAnimationFrame(() => {
            cards.scrollLeft = cards.scrollWidth;
        });
    });

    const remove = document.createElement("button");
    remove.textContent = "− Remove Last";
    remove.addEventListener("click", (e) => {
        e.preventDefault();
        if (runtime.state.clips.length <= 1) return;
        runtime.state.clips.pop();
        updateHidden(node, runtime);
        render(node, runtime);
    });

    const saveProjectButton = document.createElement("button");
    saveProjectButton.textContent = "Save Project";
    saveProjectButton.title = "Save settings + disk cache as a portable .ext project";
    saveProjectButton.addEventListener("click", (e) => {
        e.preventDefault();
        saveProject(node, runtime);
    });

    const loadProjectButton = document.createElement("button");
    loadProjectButton.textContent = "Load Project";
    loadProjectButton.title = "Load a .ext project into this Extender node";

    const projectFileInput = document.createElement("input");
    projectFileInput.type = "file";
    projectFileInput.accept = ".ext,application/zip,application/octet-stream";
    projectFileInput.style.display = "none";
    projectFileInput.addEventListener("change", async () => {
        const file = projectFileInput.files?.[0];
        projectFileInput.value = "";
        if (file) await loadProjectFile(node, runtime, file);
    });
    loadProjectButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (projectBusy(runtime)) {
            alert("Wait for the current clip generation to finish before loading a project.");
            return;
        }
        projectFileInput.click();
    });

    const counter = document.createElement("span");
    counter.style.fontSize = "11px";
    counter.style.opacity = ".8";

    const status = document.createElement("span");
    status.style.fontSize = "11px";
    status.style.opacity = ".72";
    status.style.marginLeft = "auto";
    status.style.whiteSpace = "nowrap";
    status.style.overflow = "hidden";
    status.style.textOverflow = "ellipsis";
    status.style.maxWidth = "55%";

    toolbar.append(add, remove, saveProjectButton, loadProjectButton, counter, status, projectFileInput);

    const refFileInput = document.createElement("input");
    refFileInput.type = "file";
    refFileInput.accept = "image/*,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff";
    refFileInput.style.display = "none";

    const refsSection = document.createElement("div");
    refsSection.style.height = `${REF_SECTION_HEIGHT}px`;
    refsSection.style.minWidth = "0";
    refsSection.style.flex = `0 0 ${REF_SECTION_HEIGHT}px`;
    refsSection.style.boxSizing = "border-box";
    refsSection.style.marginBottom = "7px";

    const refsHeader = document.createElement("div");
    refsHeader.textContent = "REFERENCE IMAGES — double-click a thumbnail to edit";
    refsHeader.style.fontSize = "10px";
    refsHeader.style.fontWeight = "600";
    refsHeader.style.opacity = ".75";
    refsHeader.style.height = "13px";
    refsHeader.style.lineHeight = "13px";
    refsHeader.style.marginBottom = "1px";

    const refsRow = document.createElement("div");
    refsRow.style.display = "flex";
    refsRow.style.flexDirection = "row";
    refsRow.style.width = "100%";
    refsRow.style.maxWidth = "100%";
    refsRow.style.minWidth = "0";
    refsRow.style.gap = "7px";
    refsRow.style.overflowX = "auto";
    refsRow.style.overflowY = "hidden";
    refsRow.style.paddingBottom = `${REF_SCROLLBAR_SPACE}px`;
    refsRow.style.boxSizing = "border-box";
    refsRow.style.height = `${REF_SECTION_HEIGHT - 14}px`;
    refsRow.style.scrollbarGutter = "stable";
    refsSection.append(refsHeader, refsRow);

    const cards = document.createElement("div");
    cards.style.display = "flex";
    cards.style.minWidth = "0";
    cards.style.flexDirection = "row";
    cards.style.gap = "9px";
    cards.style.overflowX = "auto";
    cards.style.overflowY = "hidden";
    cards.style.padding = `0 0 ${CARD_SCROLLBAR_SPACE}px 0`;
    cards.style.scrollbarGutter = "stable";
    cards.style.boxSizing = "border-box";
    cards.style.scrollBehavior = "smooth";
    cards.style.height = `${UI_MIN_HEIGHT - 55 - REF_SECTION_HEIGHT}px`;
    cards.style.minHeight = `${NODES2_CARDS_MIN_HEIGHT}px`;

    root.append(toolbar, refsSection, cards, refFileInput);

    const restoredValidatedPrefix = validatedPrefixFromState(state);
    const runtime = {
        state,
        jsonWidget,
        refsState,
        refsWidget,
        root,
        toolbar,
        refsSection,
        refsRow,
        cards,
        counter,
        status,
        saveProjectButton,
        loadProjectButton,
        projectFileInput,
        refFileInput,
        pendingRefSlot: -1,
        refBusy: false,
        projectOperationBusy: false,
        projectName: String(node?.properties?.h3_project_name || ""),
        domWidget: null,
        domHeight: UI_MIN_HEIGHT,
        syncingDomHeight: false,
        lastRenderMode: null,
        legacyNodeHeight: null,
        // clips_json already preserves the validated flags. Seed the visual state
        // immediately, then replace it with the authoritative disk manifest below.
        cachedCount: restoredValidatedPrefix,
        validatedCount: restoredValidatedPrefix,
        statusText: restoredValidatedPrefix
            ? `Restoring cache | validated ${restoredValidatedPrefix}`
            : "Ready",
        activeClipIndex: -1,
        activePhase: "idle",
        cacheStateRequestRunning: false,
        cacheStateRestored: false,
        expectedResolution: null,
        resolvedWidth: 0,
        resolvedHeight: 0,
        resolutionGuide: "",
        guideSourceWidth: 0,
        guideSourceHeight: 0,
        resolutionFallback: false,
        resolutionMismatch: false,
        manualWidth: Number(node?.properties?.h3_manual_width || getWidget(node, "width")?.value || 896),
        manualHeight: Number(node?.properties?.h3_manual_height || getWidget(node, "height")?.value || 576),
        applyingResolutionMirror: false,
        resolutionMirrorActive: false,
        resolutionCallbacksInstalled: false,
        // True only after an explicit .ext Load has imposed its archived
        // geometry. Any user resolution edit clears it; editing megapixels
        // also switches straight back to Auto because MP has no Manual meaning.
        projectResolutionLoaded: false,
        // True after a live resolution change has made the on-disk cache stale.
        // The backend clears/rebuilds that cache on the next Queue.
        resolutionInvalidated: false,
        ready: false,
    };

    refFileInput.addEventListener("change", async () => {
        const file = refFileInput.files?.[0];
        const slot = Number(runtime.pendingRefSlot);
        refFileInput.value = "";
        runtime.pendingRefSlot = -1;
        if (file && Number.isInteger(slot) && slot >= 0 && slot < MAX_IMAGE_REFS) {
            await uploadReference(node, runtime, slot, file);
        }
    });

    const domWidget = node.addDOMWidget("h3_extender_timeline", "timeline", root, {
        serialize: false,
        hideOnZoom: false,
        // DOMWidgetImpl.computeLayoutSize() is the official size contract.
        // Give Nodes 2.0 a little more intrinsic room, while keeping the old
        // Legacy minimum unchanged.
        getMinHeight: () =>
            globalThis.LiteGraph?.vueNodesMode ? NODES2_MIN_HEIGHT : UI_MIN_HEIGHT,
        getHeight: () => runtime.domHeight,
        afterResize: (resizedNode) => {
            const mode = domWidgetRenderMode(root);
            if (mode === "nodes2") {
                // Re-assert only intrinsic CSS. Never derive anything from
                // node.size while Vue is resolving its grid.
                root.style.height = "auto";
                root.style.minHeight = `${NODES2_MIN_HEIGHT}px`;
                root.style.setProperty("--comfy-widget-min-height", `${NODES2_MIN_HEIGHT}px`);
                root.style.maxHeight = "none";
                root.style.flex = "1 1 auto";
                root.style.paddingTop = `${5 + NODES2_TOP_GAP}px`;
                root.style.overflow = "visible";
                cards.style.height = "auto";
                cards.style.flex = "1 1 auto";
                cards.style.minHeight = `${NODES2_CARDS_MIN_HEIGHT}px`;
                runtime.lastRenderMode = "nodes2";
            } else if (mode === "legacy") {
                requestAnimationFrame(() => syncDomHeight(resizedNode, runtime, false));
            } else {
                requestAnimationFrame(() => syncDomHeight(resizedNode, runtime, false));
            }
        },
    });
    runtime.domWidget = domWidget;
    node.__h3Extender = runtime;

    installInvalidationHooks(node, runtime);
    wrapResolutionWidgetCallbacks(node, runtime);
    render(node, runtime);

    const oldConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        if (oldConfigure) oldConfigure.apply(this, arguments);

        // Workflow widget arrays are positional. The two v14.25 resolution
        // widgets were intentionally appended after clips_json so old values do
        // not shift. If this is an older workflow, force Manual to preserve its
        // historical width/height behavior. Newly-created nodes default to Auto.
        const savedWidgetValues = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        const hasSavedResolutionMode = Boolean(
            savedWidgetValues?.some((value) => value === "auto_from_ref" || value === "manual")
        );
        if (savedWidgetValues && !hasSavedResolutionMode) {
            setWidgetValue(this, "resolution_mode", "manual");
        }

        requestAnimationFrame(() => {
            const removedLegacyRefs = removeLegacyImageRefInputs(this);
            runtime.state = parseState(runtime.jsonWidget.value);
            runtime.refsState = parseRefsState(runtime.refsWidget.value);
            updateRefsHidden(this, runtime);
            const restoredValidatedPrefix = validatedPrefixFromState(runtime.state);
            runtime.cachedCount = restoredValidatedPrefix;
            runtime.validatedCount = restoredValidatedPrefix;
            if (removedLegacyRefs && refCount(runtime) === 0) {
                runtime.statusText = "Legacy image-ref sockets removed — load references in the Extender";
            }
            if (String(getWidget(this, "resolution_mode")?.value || "manual") === "manual") {
                rememberManualResolution(
                    this,
                    runtime,
                    Number(getWidget(this, "width")?.value || runtime.manualWidth || 896),
                    Number(getWidget(this, "height")?.value || runtime.manualHeight || 576),
                );
            }
            render(this, runtime);
            restoreCacheState(this, runtime);
            syncResolutionMirror(this, runtime);
            syncDomHeight(this, runtime, true);
        });
    };

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            removeLegacyImageRefInputs(node);
            runtime.ready = true;
            restoreCacheState(node, runtime);
            syncResolutionMirror(node, runtime);
            syncDomHeight(node, runtime, true);
        });
    });

    return runtime;
}


function findExtenderNodeByExecutionId(nodeId) {
    const graph = app.graph;
    if (!graph) return null;

    const wanted = String(nodeId);
    for (const node of graph._nodes || []) {
        if (
            String(node?.id) === wanted &&
            (node?.comfyClass === TARGET || node?.type === TARGET)
        ) {
            return node;
        }
    }
    return null;
}

function scrollActiveCard(runtime, index) {
    if (!runtime?.cards || index < 0) return;
    const card = runtime.cards.querySelector(
        `[data-clip-index="${index}"]`
    );
    if (!card) return;

    const left = Math.max(
        0,
        card.offsetLeft -
            Math.max(0, (runtime.cards.clientWidth - card.offsetWidth) / 2)
    );
    runtime.cards.scrollTo({
        left,
        behavior: "smooth",
    });
}

// A cancelled/failed ComfyUI execution does not call this node's onExecuted
// callback. Without an explicit terminal-event reset, the last custom progress
// event (usually "sampling") leaves the active card permanently blue until a
// page refresh. ComfyUI exposes official execution_interrupted/error/success
// websocket events, so clear only the transient rendering state when a prompt
// terminates. Cache/validation/card data are deliberately left untouched.
function clearTransientRenderingState(statusText = null) {
    const graph = app.graph;
    if (!graph) return;

    for (const node of graph._nodes || []) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;

        const runtime = node.__h3Extender;
        if (!runtime) continue;

        const wasActive =
            Number(runtime.activeClipIndex) >= 0 ||
            ["preparing", "sampling", "complete"].includes(
                String(runtime.activePhase || "")
            );
        if (!wasActive) continue;

        runtime.activeClipIndex = -1;
        runtime.activePhase = "idle";
        if (statusText) runtime.statusText = statusText;

        render(node, runtime);
        node.graph?.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "MiniMaxH3.Extender",

    setup() {
        // Official ComfyUI terminal execution events. In particular, pressing
        // Kill/Interrupt raises execution_interrupted and bypasses onExecuted.
        api.addEventListener("execution_interrupted", () => {
            clearTransientRenderingState("Rendering interrupted");
        });
        api.addEventListener("execution_error", () => {
            clearTransientRenderingState("Execution stopped by error");
        });
        // Defensive cleanup: a successful prompt should never leave a stale
        // rendering highlight even if another frontend/backend change prevents
        // the expected node UI callback from arriving.
        api.addEventListener("execution_success", () => {
            clearTransientRenderingState();
        });

        api.addEventListener(PROMPT_PACK_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime || !detail?.clips_json) return;

            runtime.jsonWidget.value = String(detail.clips_json);
            runtime.state = parseState(detail.clips_json);
            const count = Number(detail?.prompt_count || runtime.state.clips.length || 0);
            const source = String(detail?.source || "External prompt pack");
            runtime.statusText = `${source}: imported ${count} prompt${count === 1 ? "" : "s"} → ${count} clip${count === 1 ? "" : "s"}`;
            updateHidden(node, runtime);
            render(node, runtime);
            syncDomHeight(node, runtime, false);
            node.graph?.setDirtyCanvas(true, true);
        });

        api.addEventListener(REF_PACK_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime || !detail?.refs_json) return;

            runtime.refsWidget.value = String(detail.refs_json);
            runtime.refsState = parseRefsState(detail.refs_json);
            const slots = Array.isArray(detail?.imported_slots)
                ? detail.imported_slots.map((value) => Number(value)).filter((value) => Number.isInteger(value) && value >= 1 && value <= MAX_IMAGE_REFS)
                : [];
            const source = String(detail?.source || "External reference pack");
            runtime.statusText = slots.length
                ? `${source}: imported Ref ${slots.join(", ")} into internal slots`
                : `${source}: synchronized`;
            render(node, runtime);
            node.graph?.setDirtyCanvas(true, true);
        });

        api.addEventListener(PROGRESS_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime) return;

            const index = Number(detail?.clip_index ?? -1);
            runtime.activeClipIndex = Number.isFinite(index) ? index : -1;
            runtime.activePhase = String(detail?.phase || "idle");
            runtime.statusText = String(detail?.message || runtime.statusText || "Ready");

            render(node, runtime);

            if (runtime.activeClipIndex >= 0) {
                requestAnimationFrame(() => {
                    scrollActiveCard(runtime, runtime.activeClipIndex);
                });
            }

            node.graph?.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = oldCreated ? oldCreated.apply(this, arguments) : undefined;

            // New nodes must start in Auto resolution mode. Older workflows are
            // still migrated to Manual later in onConfigure when they do not
            // contain the v14.25+ resolution widgets.
            setWidgetValue(this, "resolution_mode", "auto_from_ref");

            const runtime = buildUi(this);
            removeLegacyImageRefInputs(this);
            if (runtime) {
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => syncDomHeight(this, runtime, true));
                });
            }
            return r;
        };

        const oldExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (oldExecuted) oldExecuted.apply(this, arguments);
            const runtime = buildUi(this);
            if (!runtime) return;

            const info = message?.h3_extender_state?.[0];
            if (!info) return;

            if (info.clips_json) {
                runtime.jsonWidget.value = info.clips_json;
                runtime.state = parseState(info.clips_json);
            }
            if (info.refs_json) {
                runtime.refsWidget.value = info.refs_json;
                runtime.refsState = parseRefsState(info.refs_json);
            }
    
            const generated = Array.isArray(info.generated) ? info.generated : [];
            for (const humanIndex of generated) {
                const i = Number(humanIndex) - 1;
                const clip = runtime.state.clips[i];
                // Only prepare a next seed for a candidate. A validated cached
                // clip is never touched by this automatic seed behavior.
                if (clip && !clip.validated) {
                    advanceSeedAfterGenerate(clip);
                }
            }

            // Critical: persist the next seed into clips_json. This changes the
            // node input hash, so pressing Queue again really re-executes it.
            if (generated.length) {
                updateHidden(this, runtime);
            }

            runtime.cachedCount = Number(info.cached_count || 0);
            runtime.validatedCount = Number(info.validated_count || 0);
            runtime.resolvedWidth = Number(info.resolved_width || 0);
            runtime.resolvedHeight = Number(info.resolved_height || 0);
            runtime.resolutionGuide = String(info.resolution_guide || "");
            runtime.guideSourceWidth = Number(info.resolution_guide_width || 0);
            runtime.guideSourceHeight = Number(info.resolution_guide_height || 0);
            runtime.resolutionFallback = Boolean(info.resolution_fallback);
            runtime.resolutionMismatch = Boolean(info.resolution_mismatch);
            if (runtime.resolvedWidth > 0 && runtime.resolvedHeight > 0) {
                // Backend execution is authoritative. After a resolution-change
                // run, this becomes the new baseline for future invalidation.
                runtime.expectedResolution = {
                    width: runtime.resolvedWidth,
                    height: runtime.resolvedHeight,
                };
                runtime.resolutionInvalidated = false;
            }
            runtime.activeClipIndex = -1;
            runtime.activePhase = "idle";
            runtime.statusText = String(info.status || "Ready");
            if (runtime.resolutionMismatch && Number(info.cache_width || 0) > 0) {
                runtime.statusText +=
                    ` | WARNING cache ${Number(info.cache_width)}x${Number(info.cache_height)} differs`;
            }
            syncResolutionMirror(this, runtime);
            render(this, runtime);
            syncDomHeight(this, runtime, false);
        };
    },
});
