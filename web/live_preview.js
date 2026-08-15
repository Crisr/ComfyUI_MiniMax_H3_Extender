import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3MotionContextDiskFinalDecode";
const DISK_JOIN_TARGET = "MiniMaxH3MotionContextDiskJoin";
const EXTENDER_TARGET = "MiniMaxH3Extender";

function stripFinalDecodeOutputs(node) {
    if (!node?.outputs?.length) return;

    // Old workflows serialize the nine legacy outputs in the graph JSON.
    // RETURN_TYPES=() prevents them on newly-defined nodes, but LiteGraph can
    // restore the serialized sockets during configure(). Remove them here too.
    while (node.outputs?.length) {
        const index = node.outputs.length - 1;
        if (typeof node.removeOutput === "function") {
            node.removeOutput(index);
        } else {
            // Fallback for older LiteGraph builds. These outputs were never
            // meant to be consumed; remove the visual slots at minimum.
            node.outputs.splice(index, 1);
        }
    }

    node.graph?.setDirtyCanvas(true, true);
}

function isDiskJoin(node) {
    return (
        node?.comfyClass === DISK_JOIN_TARGET ||
        node?.type === DISK_JOIN_TARGET
    );
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function isFalseValue(value) {
    return value === false || value === 0 || value === "false";
}

function setValidatedFalse(node) {
    const widget = getWidget(node, "validated");
    if (!widget) return false;

    if (!isFalseValue(widget.value)) {
        // Do NOT invoke its callback here: the recursive graph traversal below
        // already propagates invalidation and avoids callback recursion.
        widget.value = false;
        node.graph?.setDirtyCanvas(true, true);
        return true;
    }
    return false;
}

function downstreamDiskJoins(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return [];

    const output = node?.outputs?.find((o) => o?.name === "cache");
    const links = output?.links || [];
    const result = [];

    for (const linkId of links) {
        const link = graph.links?.[linkId];
        if (!link) continue;

        const target = graph.getNodeById?.(link.target_id);
        if (target && isDiskJoin(target)) {
            result.push(target);
        }
    }
    return result;
}

function invalidateDownstream(node) {
    const visited = new Set();

    function walk(current) {
        for (const next of downstreamDiskJoins(current)) {
            const key = next.id ?? next;
            if (visited.has(key)) continue;
            visited.add(key);

            setValidatedFalse(next);
            walk(next);
        }
    }

    walk(node);
    node?.graph?.setDirtyCanvas(true, true);
}

function installValidationCascade(node) {
    if (!node || node.__h3ValidationCascadeInstalled) return;

    const widget = getWidget(node, "validated");
    if (!widget) {
        // Widgets can finish materializing just after onNodeCreated.
        requestAnimationFrame(() => installValidationCascade(node));
        return;
    }

    const originalCallback = widget.callback;

    widget.callback = function (value) {
        const result = originalCallback
            ? originalCallback.apply(this, arguments)
            : undefined;

        // If clip N is invalidated, clips N+1... are no longer valid because
        // their Motion Context depended on the old version of clip N.
        if (isFalseValue(value)) {
            invalidateDownstream(node);
        }

        return result;
    };

    node.__h3ValidationCascadeInstalled = true;
}


const PLAYER_MIN_WIDTH = 380;
const PLAYER_MIN_HEIGHT = 220;
const LABEL_HEIGHT = 22;
const BOTTOM_PAD = 14;

function mediaUrl(info) {
    const params = new URLSearchParams();
    params.set("filename", info.filename || "");
    params.set("type", info.type || "temp");
    params.set("subfolder", info.subfolder || "");
    return api.apiURL("/view?" + params.toString());
}


function findUpstreamExtenderId(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return null;

    const cacheInput = (node.inputs || []).find((input) => input?.name === "cache");
    const linkId = cacheInput?.link;
    if (linkId == null) return null;

    const link = graph.links?.[linkId];
    if (!link) return null;

    const origin = graph.getNodeById?.(link.origin_id)
        || (graph._nodes || []).find((n) => String(n?.id) === String(link.origin_id));
    if (!origin) return null;

    if (
        origin?.comfyClass === EXTENDER_TARGET ||
        origin?.type === EXTENDER_TARGET
    ) {
        return origin.id;
    }

    return null;
}

async function restorePreviewOnLoad(node, state, attempt = 0) {
    if (!node || !state || state.liveLoaded || state.restoreLoaded) return;

    const ownerId = findUpstreamExtenderId(node);
    if (ownerId == null) {
        // Workflow links may be restored a little after node.configure().
        if (attempt < 12) {
            setTimeout(() => restorePreviewOnLoad(node, state, attempt + 1), 80);
        }
        return;
    }

    if (state.restoreRequestRunning) return;
    state.restoreRequestRunning = true;

    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(ownerId));
        params.set("final_id", String(node.id));

        const response = await fetch(
            api.apiURL("/h3_extender/restored_preview?" + params.toString())
        );
        if (!response.ok) return;

        const payload = await response.json();
        if (!payload?.found || !payload?.video?.filename) return;
        if (state.liveLoaded) return;

        const clips = Number(payload.clip_count || 0);
        const frames = Number(payload.frame_count || 0);
        state.label.textContent =
            `RESTORED PREVIEW — ${clips} clip${clips === 1 ? "" : "s"} (${frames} frames)`;

        state.video.src = mediaUrl(payload.video) + "&t=" + Date.now();
        state.video.load();

        // Browsers can block autoplay after a page reload. The preview is still
        // immediately visible and ready; play() succeeds when policy allows it.
        state.video.play().catch(() => {});
        state.restoreLoaded = true;

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                syncPlayerToNode(node, state, true);
            });
        });
    } catch (_) {
        // Startup preview is convenience only; never break workflow loading.
    } finally {
        state.restoreRequestRunning = false;
    }
}

function syncPlayerToNode(node, state, growNodeIfNeeded = false, retry = 0) {
    if (!node || !state?.widget) return;

    const widgetY = Number(state.widget.last_y);

    // LiteGraph only knows the real widget Y after layout/draw.
    if (!Number.isFinite(widgetY) || widgetY <= 0) {
        if (retry < 5) {
            requestAnimationFrame(() =>
                syncPlayerToNode(node, state, growNodeIfNeeded, retry + 1)
            );
        }
        return;
    }

    let nodeW = Number(node.size?.[0] || PLAYER_MIN_WIDTH);
    let nodeH = Number(node.size?.[1] || 0);

    if (nodeW < PLAYER_MIN_WIDTH) {
        nodeW = PLAYER_MIN_WIDTH;
    }

    const minimumNodeH = widgetY + PLAYER_MIN_HEIGHT + BOTTOM_PAD;

    // At creation/execution ensure enough room for a useful player.
    // During manual resize, DO NOT force a fixed height: use the user's node height.
    if (growNodeIfNeeded && nodeH < minimumNodeH) {
        nodeH = minimumNodeH;
        node.setSize([nodeW, nodeH]);
    } else if (Number(node.size?.[0] || 0) < PLAYER_MIN_WIDTH) {
        node.setSize([nodeW, nodeH]);
    }

    const availableH = Math.max(
        PLAYER_MIN_HEIGHT,
        Number(node.size?.[1] || nodeH) - widgetY - BOTTOM_PAD
    );

    state.currentHeight = availableH;

    state.box.style.height = `${availableH}px`;
    state.video.style.height =
        `${Math.max(80, availableH - LABEL_HEIGHT - 4)}px`;

    // Keep the DOM widget's logical height synchronized with the actual player.
    // This makes LiteGraph clipping/hit-testing agree with what is visible.
    if (state.widget.options) {
        state.widget.options.getMinHeight = () => PLAYER_MIN_HEIGHT;
        state.widget.options.getHeight = () => state.currentHeight;
    }
    state.widget.getHeight = () => state.currentHeight;

    node.graph?.setDirtyCanvas(true, true);
}

function makePlayer(node) {
    if (node.__h3LivePreview) return node.__h3LivePreview;

    const box = document.createElement("div");
    box.style.width = "100%";
    box.style.height = `${PLAYER_MIN_HEIGHT}px`;
    box.style.boxSizing = "border-box";
    box.style.padding = "4px 0 0 0";
    box.style.background = "transparent";
    box.style.overflow = "hidden";

    const label = document.createElement("div");
    label.textContent = "FULL LIVE PREVIEW";
    label.style.fontSize = "11px";
    label.style.opacity = "0.78";
    label.style.height = `${LABEL_HEIGHT}px`;
    label.style.lineHeight = `${LABEL_HEIGHT}px`;
    label.style.margin = "0";
    label.style.whiteSpace = "nowrap";
    label.style.overflow = "hidden";
    label.style.textOverflow = "ellipsis";

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.style.display = "block";
    video.style.width = "100%";
    video.style.height = `${PLAYER_MIN_HEIGHT - LABEL_HEIGHT - 4}px`;
    video.style.objectFit = "contain";
    video.style.background = "#000";
    video.style.borderRadius = "4px";

    box.appendChild(label);
    box.appendChild(video);

    const state = {
        box,
        label,
        video,
        widget: null,
        currentHeight: PLAYER_MIN_HEIGHT,
        liveLoaded: false,
        restoreLoaded: false,
        restoreRequestRunning: false,
    };

    const widget = node.addDOMWidget("h3_live_preview", "preview", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => PLAYER_MIN_HEIGHT,
        getHeight: () => state.currentHeight,
    });
    state.widget = widget;

    node.__h3LivePreview = state;

    // User resizes node -> player immediately consumes all available height.
    const oldResize = node.onResize;
    node.onResize = function () {
        if (oldResize) oldResize.apply(this, arguments);

        requestAnimationFrame(() => {
            syncPlayerToNode(this, state, false);
        });
    };

    const oldRemove = node.onRemoved;
    node.onRemoved = function () {
        try {
            video.pause();
            video.removeAttribute("src");
            video.load();
        } catch (_) {}

        if (oldRemove) oldRemove.apply(this, arguments);
    };

    // First layout: guarantee minimum useful player size.
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            syncPlayerToNode(node, state, true);
        });
    });

    return state;
}

app.registerExtension({
    name: "MiniMaxH3.MotionContext.LivePreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === DISK_JOIN_TARGET) {
            const oldJoinCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = oldJoinCreated
                    ? oldJoinCreated.apply(this, arguments)
                    : undefined;

                installValidationCascade(this);

                // One extra frame covers workflows restored from JSON where
                // widgets can be populated just after node creation.
                requestAnimationFrame(() => {
                    installValidationCascade(this);
                });

                return r;
            };

            return;
        }

        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = oldCreated
                ? oldCreated.apply(this, arguments)
                : undefined;

            stripFinalDecodeOutputs(this);
            const state = makePlayer(this);

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    stripFinalDecodeOutputs(this);
                    syncPlayerToNode(this, state, true);
                    restorePreviewOnLoad(this, state);
                });
            });

            return r;
        };

        // Existing workflow JSON can restore legacy output slots after
        // onNodeCreated. Strip them again immediately after configuration.
        const oldConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = oldConfigure
                ? oldConfigure.apply(this, arguments)
                : undefined;

            stripFinalDecodeOutputs(this);
            const state = makePlayer(this);
            requestAnimationFrame(() => {
                stripFinalDecodeOutputs(this);
                restorePreviewOnLoad(this, state);
            });
            return r;
        };

        const oldExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (oldExecuted) oldExecuted.apply(this, arguments);

            const info = message?.h3_video?.[0];
            if (!info?.filename) return;

            const state = makePlayer(this);
            state.liveLoaded = true;
            const meta = message?.h3_preview_info?.[0];

            if (meta?.mode === "clip_by_clip") {
                const s = Number(meta.seam_shift || 0);
                state.label.textContent =
                    `FULL LIVE PREVIEW — ${meta.total_clips} clip${meta.total_clips > 1 ? "s" : ""} — shift ${s >= 0 ? "+" : ""}${s}`;
            } else if (meta?.mode === "full_batch") {
                state.label.textContent =
                    `FINAL PREVIEW — ${meta.total_clips} clips (${meta.preview_frames} frames)`;
            } else {
                state.label.textContent = "FULL LIVE PREVIEW";
            }

            state.video.src = mediaUrl(info) + "&t=" + Date.now();
            state.video.load();
            state.video.play().catch(() => {});

            // Do not reset a node the user already enlarged.
            // Only enforce the minimum if necessary, then fit the player
            // to the CURRENT node height.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    syncPlayerToNode(this, state, true);
                });
            });
        };
    },
});
