import { app } from "../../scripts/app.js";

const TARGET = "MiniMaxH3ReferencePackBridge";
const MAX_REFS = 9;
const REF_RE = /^ref_([1-9])$/;

function refIndex(input) {
    const match = String(input?.name || "").match(REF_RE);
    if (!match) return 0;
    const index = Number(match[1]);
    return Number.isInteger(index) && index >= 1 && index <= MAX_REFS ? index : 0;
}

function isConnected(input) {
    return input?.link !== null && input?.link !== undefined;
}

function currentRefInputs(node) {
    return (node?.inputs || [])
        .map((input, slot) => ({ input, slot, index: refIndex(input) }))
        .filter((entry) => entry.index > 0)
        .sort((a, b) => a.index - b.index);
}

function addRefInput(node, index) {
    if (!node || index < 1 || index > MAX_REFS) return false;
    if ((node.inputs || []).some((input) => refIndex(input) === index)) return false;
    node.addInput(`ref_${index}`, "IMAGE");
    return true;
}

function removeRefInput(node, slot) {
    if (!node || slot < 0 || slot >= (node.inputs || []).length) return false;
    try {
        node.removeInput(slot);
        return true;
    } catch (_) {
        return false;
    }
}

function fitNodeHeight(node) {
    try {
        const computed = node?.computeSize?.();
        const height = Number(computed?.[1]);
        const width = Number(node?.size?.[0]);
        if (Number.isFinite(height) && height > 0 && Number.isFinite(width) && width > 0) {
            node.setSize?.([width, height]);
        }
    } catch (_) {}
}

function syncRefInputs(node) {
    if (!node || node.__h3RefBridgeSyncing) return;
    node.__h3RefBridgeSyncing = true;

    let changed = false;
    try {
        let entries = currentRefInputs(node);
        const connected = entries.filter(({ input }) => isConnected(input));
        const highestConnected = connected.reduce((max, entry) => Math.max(max, entry.index), 0);

        // Ref numbers are identities. Keep every slot through the highest
        // connected reference, including holes, plus exactly one free next slot.
        // This means ref_3 never becomes ref_2 when ref_2 is disconnected.
        const desiredLast = highestConnected > 0
            ? Math.min(MAX_REFS, highestConnected + 1)
            : 1;

        // Remove only trailing unused sockets. Never remove a middle hole because
        // doing so would shift the logical Ref number of a connected cable.
        const trailing = entries
            .filter(({ index }) => index > desiredLast)
            .sort((a, b) => b.slot - a.slot);
        for (const { slot } of trailing) {
            changed = removeRefInput(node, slot) || changed;
        }

        // Normal creation/configuration starts from the backend's complete 1..9
        // declaration, so this mostly recreates the next autogrow socket after a
        // previous shrink. Keep names stable and never rename connected inputs.
        for (let index = 1; index <= desiredLast; index++) {
            changed = addRefInput(node, index) || changed;
        }

        if (changed) fitNodeHeight(node);
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__h3RefBridgeSyncing = false;
    }
}

function deferSync(node) {
    if (!node || node.__h3RefBridgeSyncQueued) return;
    node.__h3RefBridgeSyncQueued = true;
    requestAnimationFrame(() => {
        node.__h3RefBridgeSyncQueued = false;
        syncRefInputs(node);
    });
}

app.registerExtension({
    name: "MiniMaxH3.ReferencePackBridge.DynamicInputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = oldCreated ? oldCreated.apply(this, arguments) : undefined;
            deferSync(this);
            return result;
        };

        const oldConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
            deferSync(this);
            return result;
        };

        const oldConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = oldConnectionsChange
                ? oldConnectionsChange.apply(this, arguments)
                : undefined;
            deferSync(this);
            return result;
        };
    },
});
