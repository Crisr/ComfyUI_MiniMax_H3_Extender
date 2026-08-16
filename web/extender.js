import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3Extender";
const PROGRESS_EVENT = "h3_extender_progress";
const CARD_WIDTH = 318;
const UI_MIN_HEIGHT = 460;
const NODES2_MIN_HEIGHT = 510;
// Keep a real visual gap between the native Nodes 2.0 widgets and the CLIP
// panel. This is internal padding only: we deliberately do NOT rewrite Vue
// grid tracks or absolutely position the DOM widget.
const NODES2_TOP_GAP = 28;
const NODE_MIN_WIDTH = 980;
const BOTTOM_PAD = 16;
// Leave an empty gutter under each card so an overlay horizontal scrollbar
// never covers the Validated/footer row.
const CARD_SCROLLBAR_SPACE = 24;
const CARD_MIN_HEIGHT = 355;
const NODES2_CARDS_MIN_HEIGHT = CARD_MIN_HEIGHT + CARD_SCROLLBAR_SPACE;
const MAX_IMAGE_REFS = 9;

// Validation is intentionally explicit. Generation parameters and input
// connections may be changed at any time without touching already validated
// clips. To regenerate an accepted clip, the user explicitly unchecks its
// Validated box; that is the ONLY action that invalidates that clip and every
// dependent clip after it.


function isRefInputName(name) {
    return /^ref_\d+$/.test(String(name || ""));
}

function refInputNumber(name) {
    const m = /^ref_(\d+)$/.exec(String(name || ""));
    return m ? Number(m[1]) : NaN;
}

function refInputConnected(input) {
    return input && input.link != null;
}

function findInputIndexByName(node, name) {
    return (node?.inputs || []).findIndex((input) => input?.name === name);
}

function getOriginNodeForLink(node, linkId) {
    const graph = node?.graph;
    const link = graph?.links?.[linkId];
    if (!graph || !link) return null;
    if (typeof graph.getNodeById === "function") return graph.getNodeById(link.origin_id);
    return (graph._nodes || []).find((n) => String(n?.id) === String(link.origin_id)) || null;
}

function moveRefConnection(node, fromNum, toNum) {
    if (!node || fromNum === toNum) return false;

    const fromIdx = findInputIndexByName(node, `ref_${fromNum}`);
    const toIdx = findInputIndexByName(node, `ref_${toNum}`);
    if (fromIdx < 0 || toIdx < 0) return false;

    const fromInput = node.inputs?.[fromIdx];
    if (!refInputConnected(fromInput)) return false;

    const linkId = fromInput.link;
    const originNode = getOriginNodeForLink(node, linkId);
    const link = node?.graph?.links?.[linkId];
    const originSlot = link?.origin_slot;
    if (!originNode || originSlot == null) return false;

    // Recreate the link on the earlier socket, then drop the old one.
    if (refInputConnected(node.inputs?.[toIdx])) {
        node.disconnectInput(toIdx);
    }

    const created = originNode.connect(originSlot, node, toIdx);
    if (!created) return false;

    // If the original connection still remains on the old slot, remove it now.
    if (refInputConnected(node.inputs?.[fromIdx])) {
        node.disconnectInput(fromIdx);
    }
    return true;
}

function compactRefInputConnections(node) {
    if (!node || node.__h3RefCompacting) return;

    node.__h3RefCompacting = true;
    try {
        // Ensure the full range exists temporarily so we can target early slots safely.
        for (let i = 1; i <= MAX_IMAGE_REFS; i++) {
            if (findInputIndexByName(node, `ref_${i}`) < 0) {
                node.addInput(`ref_${i}`, "IMAGE", { forceInput: true });
            }
        }

        let targetNum = 1;
        for (let sourceNum = 1; sourceNum <= MAX_IMAGE_REFS; sourceNum++) {
            const sourceIdx = findInputIndexByName(node, `ref_${sourceNum}`);
            const sourceInput = sourceIdx >= 0 ? node.inputs?.[sourceIdx] : null;
            if (!refInputConnected(sourceInput)) continue;
            if (sourceNum !== targetNum) {
                moveRefConnection(node, sourceNum, targetNum);
            }
            targetNum += 1;
        }
    } finally {
        node.__h3RefCompacting = false;
    }
}

function syncDynamicRefInputs(node) {
    if (!node) return;

    // When the backend creates the node, all ref_1..ref_9 sockets exist.
    // Keep only the connected prefix plus exactly one empty socket.
    let highestConnected = 0;
    for (const input of node.inputs || []) {
        if (isRefInputName(input?.name) && refInputConnected(input)) {
            highestConnected = Math.max(highestConnected, refInputNumber(input.name));
        }
    }

    const wanted = Math.max(1, Math.min(MAX_IMAGE_REFS, highestConnected + 1));

    // Remove trailing unneeded ref sockets from the end first.
    for (let i = MAX_IMAGE_REFS; i > wanted; i--) {
        const idx = findInputIndexByName(node, `ref_${i}`);
        if (idx >= 0) {
            const input = node.inputs[idx];
            if (!refInputConnected(input)) {
                node.removeInput(idx);
            }
        }
    }

    // Ensure the visible contiguous range exists.
    for (let i = 1; i <= wanted; i++) {
        if (findInputIndexByName(node, `ref_${i}`) < 0) {
            node.addInput(`ref_${i}`, "IMAGE", { forceInput: true });
        }
    }

    node.graph?.setDirtyCanvas(true, true);
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

function newClip(index) {
    return {
        id: `clip_${index + 1}_${Date.now().toString(36)}`,
        prompt: "",
        seed: randomSeed(),
        seed_mode: "randomize",
        duration: 10.0,
        validated: false,
    };
}

function parseState(raw) {
    try {
        const p = JSON.parse(raw || "{}");
        const clips = Array.isArray(p) ? p : p?.clips;
        if (Array.isArray(clips) && clips.length) {
            return {
                version: 1,
                clips: clips.map((c, i) => ({
                    id: String(c?.id || `clip_${i + 1}`),
                    prompt: String(c?.prompt || ""),
                    seed: Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Number(c?.seed || 0))),
                    seed_mode: ["randomize", "fixed", "increment", "decrement"].includes(String(c?.seed_mode))
                        ? String(c.seed_mode)
                        : "randomize",
                    duration: Math.max(0.25, Math.min(150, Number(c?.duration || 10))),
                    validated: Boolean(c?.validated),
                })),
            };
        }
    } catch (_) {}
    return { version: 1, clips: [newClip(0)] };
}

function serializeState(state) {
    return JSON.stringify({ version: 1, clips: state.clips });
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
        runtime.cacheStateRestored = true;
        runtime.statusText =
            `Restored cache | cached ${runtime.cachedCount}/${runtime.state.clips.length} | ` +
            `validated ${runtime.validatedCount}`;
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

// Nodes 2.0 (Vue) can render the native multiline STRING row before our
// onNodeCreated code gets a chance to touch the widget object. Hide that row
// pre-emptively with CSS, using the same proven strategy as ComfyUI_Stem_Mixer.
// MiniMaxH3Extender has a single native textarea: clips_json.
(function injectClipsJsonHideRule() {
    if (document.getElementById("h3-extender-hide-clips-json")) return;
    const style = document.createElement("style");
    style.id = "h3-extender-hide-clips-json";
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

function render(node, runtime) {
    const { state, cards, counter, status } = runtime;
    cards.replaceChildren();

    counter.textContent = `${state.clips.length} clip${state.clips.length > 1 ? "s" : ""}`;
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
        title.style.flex = "1";

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
            st === "validated" ? "✓ VALIDATED" :
            st === "candidate" ? "● CANDIDATE" :
            st === "current" ? "● NEXT" :
            st === "cached" ? "CACHE" : "○";

        head.append(title, badge);
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
        runtime.cards.style.height = `${Math.max(340, available - 55)}px`;
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
    // No parameter/input change is allowed to alter clip validation. The only
    // special connection handling kept here is the dynamic ref socket UI.
    const oldConnections = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, linkInfo, inputInfo) {
        if (oldConnections) oldConnections.apply(this, arguments);

        const slot = this.inputs?.[index];
        const slotName = slot?.name;

        if (type === 1 && isRefInputName(slotName) && !this.__h3RefCompacting) {
            compactRefInputConnections(this);
            syncDynamicRefInputs(this);
        }
    };
}


function buildUi(node) {
    if (node.__h3Extender) return node.__h3Extender;

    const jsonWidget = getWidget(node, "clips_json");
    if (!jsonWidget) return null;
    hideNativeWidget(node, jsonWidget);

    const state = parseState(jsonWidget.value);

    const root = document.createElement("div");
    root.style.width = "100%";
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

    toolbar.append(add, remove, counter, status);

    const cards = document.createElement("div");
    cards.style.display = "flex";
    cards.style.flexDirection = "row";
    cards.style.gap = "9px";
    cards.style.overflowX = "auto";
    cards.style.overflowY = "hidden";
    cards.style.padding = `0 0 ${CARD_SCROLLBAR_SPACE}px 0`;
    cards.style.scrollbarGutter = "stable";
    cards.style.boxSizing = "border-box";
    cards.style.scrollBehavior = "smooth";
    cards.style.height = `${UI_MIN_HEIGHT - 55}px`;
    cards.style.minHeight = `${NODES2_CARDS_MIN_HEIGHT}px`;

    root.append(toolbar, cards);

    const restoredValidatedPrefix = validatedPrefixFromState(state);
    const runtime = {
        state,
        jsonWidget,
        root,
        toolbar,
        cards,
        counter,
        status,
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
        ready: false,
    };

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
    render(node, runtime);

    const oldConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        if (oldConfigure) oldConfigure.apply(this, arguments);
        requestAnimationFrame(() => {
            compactRefInputConnections(this);
            syncDynamicRefInputs(this);
            runtime.state = parseState(runtime.jsonWidget.value);
            const restoredValidatedPrefix = validatedPrefixFromState(runtime.state);
            runtime.cachedCount = restoredValidatedPrefix;
            runtime.validatedCount = restoredValidatedPrefix;
            render(this, runtime);
            restoreCacheState(this, runtime);
            syncDomHeight(this, runtime, true);
        });
    };

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            compactRefInputConnections(node);
            syncDynamicRefInputs(node);
            runtime.ready = true;
            restoreCacheState(node, runtime);
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

app.registerExtension({
    name: "MiniMaxH3.Extender",

    setup() {
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
            const runtime = buildUi(this);
            compactRefInputConnections(this);
            syncDynamicRefInputs(this);
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
            runtime.activeClipIndex = -1;
            runtime.activePhase = "idle";
            runtime.statusText = String(info.status || "Ready");
            render(this, runtime);
            syncDomHeight(this, runtime, false);
        };
    },
});
