import { app } from "../../scripts/app.js";

const NODE_ID = "AlphaAnythingElementPrompts";
const FIELDS = ["elements", "background"];
const TEMPLATE_WIDGET = "template";
const TEMPLATE_ROWS = 6;

let templateHeight = 0;

function fiveRowHeight() {
    if (templateHeight) return templateHeight;
    const probe = document.createElement("textarea");
    probe.className = "comfy-multiline-input";
    probe.rows = TEMPLATE_ROWS;
    probe.style.cssText = "position:absolute;left:-9999px;top:0;visibility:hidden";
    document.body.append(probe);
    templateHeight = probe.offsetHeight || probe.scrollHeight || 64;
    probe.remove();
    return templateHeight;
}

app.registerExtension({
    name: "AlphaAnything.ElementPromptsFill",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, []);
            const widget = this.widgets?.find((w) => w.name === TEMPLATE_WIDGET);
            if (!widget) return;
            const height = fiveRowHeight();
            // widget.options.getMinHeight = () => height;
            widget.options.getMaxHeight = () => height;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, [message]);
            if (!message) return;

            let changed = false;
            for (const field of FIELDS) {
                const values = message[field];
                if (!Array.isArray(values) || typeof values[0] !== "string") continue;
                const widget = this.widgets?.find((w) => w.name === field);
                if (!widget || widget.value === values[0]) continue;
                widget.value = values[0];
                changed = true;
            }

            if (changed) this.setDirtyCanvas?.(true, true);
        };
    },
});
