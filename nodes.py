from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .alpha_branch import decode_rgba

ADAPTER_FOLDER = "alphaanything"
AlphaAnythingAdapter = io.Custom("ALPHA_ANYTHING_ADAPTER")

_ADAPTER_DIR = os.path.join(folder_paths.models_dir, ADAPTER_FOLDER)
os.makedirs(_ADAPTER_DIR, exist_ok=True)
folder_paths.add_model_folder_path(ADAPTER_FOLDER, _ADAPTER_DIR)

_REPO = "WLuLi/AlphaAnything"
_MIRRORS = ("https://huggingface.co", "https://hf-mirror.com")
_ADAPTER_FILES = ("adapter_config.json", "pytorch_lora_weights.safetensors", "alpha_decoder.safetensors")
_KNOWN_ADAPTERS = (
    "flux2_klein_base_9b_i2rgba",
    "flux2_klein_base_9b_t2rgba",
    "qwen_image_2512_t2rgba",
    "qwen_image_edit_2511_i2rgba",
)


def _adapter_names() -> list[str]:
    names = set(_KNOWN_ADAPTERS)
    if os.path.isdir(_ADAPTER_DIR):
        for entry in os.scandir(_ADAPTER_DIR):
            if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "adapter_config.json")):
                names.add(entry.name)
    return sorted(names)


def _download(url: str, destination: str) -> None:
    partial = destination + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-AlphaAnything"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(partial, "wb") as file:
            total = int(response.headers.get("Content-Length") or 0)
            progress = comfy.utils.ProgressBar(total) if total else None
            written = 0
            while chunk := response.read(1 << 20):
                file.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    if total and written != total:
        os.remove(partial)
        raise RuntimeError(f"Truncated download from {url}: {written} of {total} bytes.")
    os.replace(partial, destination)


def _adapter_path(name: str) -> str:
    if name in (".", "..") or os.path.basename(name) != name or name not in _adapter_names():
        raise ValueError(f"Invalid AlphaAnything adapter name {name!r}.")
    path = os.path.join(_ADAPTER_DIR, name)
    os.makedirs(path, exist_ok=True)
    for filename in _ADAPTER_FILES:
        destination = os.path.join(path, filename)
        if os.path.isfile(destination):
            continue
        errors = []
        for mirror in _MIRRORS:
            url = f"{mirror}/{_REPO}/resolve/main/{urllib.parse.quote(name)}/{urllib.parse.quote(filename)}"
            logging.info(f"Downloading AlphaAnything {name}/{filename} from {mirror}")
            try:
                _download(url, destination)
                break
            except OSError as error:
                errors.append(f"{mirror}: {error}")
        else:
            raise RuntimeError(f"Failed to download AlphaAnything {name}/{filename} ({'; '.join(errors)}).")
    return path


class AlphaAnythingLoRALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlphaAnythingLoRALoader",
            display_name="AlphaAnything LoRA Loader",
            category="alphaanything",
            inputs=[
                io.Combo.Input("adapter", options=_adapter_names(), tooltip="Adapter directory containing adapter_config.json."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                AlphaAnythingAdapter.Output(display_name="alpha"),
            ],
        )

    @classmethod
    def execute(cls, adapter, strength, model) -> io.NodeOutput:
        path = _adapter_path(adapter)
        with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        lora_config = config["lora"]
        # ComfyUI loads PEFT loras without their alpha, so apply alpha/rank as the base strength.
        scale = strength * float(lora_config["alpha"]) / float(lora_config["rank"])
        lora = comfy.utils.load_torch_file(os.path.join(path, "pytorch_lora_weights.safetensors"))
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, scale, 0)

        state = comfy.utils.load_torch_file(os.path.join(path, "alpha_decoder.safetensors"))
        return io.NodeOutput(model, {
            "backbone": config["backbone"],
            "config": config["vae_alpha_adapter"],
            "state": state,
        })


_ELEMENT_TEMPLATE = (
    "Extract only {element}. Preserve true natural partial transparency. "
    "Everything outside the requested element must be transparent. "
    "Do not recenter, resize, crop, restyle or add objects."
)

_ELEMENTS_SECTION = re.compile(r"<ELEMENTS>(.*?)(?:</ELEMENTS>|<BACKGROUND>|$)", re.DOTALL | re.IGNORECASE)
_BACKGROUND_SECTION = re.compile(r"<BACKGROUND>(.*?)(?:</BACKGROUND>|<ELEMENTS>|$)", re.DOTALL | re.IGNORECASE)
# Only the leading [E1] tag of a line; the same bracket text inside a description stays.
_ELEMENT_TAG = re.compile(r"^\[E\d+\]\s*")


def _split_vlm_plan(text: str) -> tuple[str, str]:
    elements = _ELEMENTS_SECTION.search(text)
    background = _BACKGROUND_SECTION.search(text)
    if elements is None and background is None:
        return text, ""
    return (elements.group(1) if elements else ""), (background.group(1) if background else "")


class AlphaAnythingElementPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlphaAnythingElementPrompts",
            display_name="AlphaAnything Element Prompts",
            category="alphaanything",
            inputs=[
                io.String.Input("vlm", optional=True, force_input=True, tooltip="VLM output with <ELEMENTS> / <BACKGROUND> sections."),
                io.Boolean.Input("use_vlm", default=True, label_on="VLM input", label_off="Manual", tooltip="Read the VLM input and split it by tags; off uses the fields below and ignores the input."),
                io.String.Input("elements", multiline=True, default="", tooltip="Element descriptions, one per line; used when the VLM input is off, empty or unconnected."),
                io.String.Input("template", multiline=True, default=_ELEMENT_TEMPLATE, tooltip="Use {element} as the placeholder; without it the element text is appended."),
                io.String.Input("background", multiline=True, default="", tooltip="Full background-layer instruction, used verbatim and placed first. Leave empty to skip."),
            ],
            outputs=[
                io.String.Output(display_name="prompts", is_output_list=True),
                io.Boolean.Output(display_name="with_alpha", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, vlm, use_vlm, elements, template, background) -> io.NodeOutput:
        from_vlm = bool(use_vlm and vlm)
        if from_vlm:
            elements, background = _split_vlm_plan(vlm)
        lines = [_ELEMENT_TAG.sub("", line.strip()).strip() for line in elements.splitlines()]
        lines = [line for line in lines if line]
        background = background.strip()
        prompts = []
        with_alpha = []
        if background:
            prompts.append(background)
            with_alpha.append(False)
        for element in lines:
            if "{element}" in template:
                prompts.append(template.replace("{element}", element))
            else:
                prompts.append(f"{template.strip()} {element}".strip())
            with_alpha.append(True)
        if not prompts:
            raise ValueError("AlphaAnything received neither elements nor a background instruction.")
        # Frontend writes these back into the widgets; blank lines there are skipped when parsing.
        ui = {"elements": ["\n\n".join(lines)], "background": [background]} if from_vlm else None
        return io.NodeOutput(prompts, with_alpha, ui=ui)


class AlphaAnythingVaeDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlphaAnythingVaeDecode",
            display_name="AlphaAnything VAE Decode (RGBA)",
            category="alphaanything",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                AlphaAnythingAdapter.Input("alpha"),
                io.Boolean.Input("with_alpha", optional=True, force_input=True, tooltip="Connect Element Prompts' with_alpha output so background layers skip the alpha branch; unconnected layers always decode alpha."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, alpha, with_alpha=None) -> io.NodeOutput:
        return io.NodeOutput(decode_rgba(vae, samples["samples"], alpha, with_alpha=with_alpha is not False))


class AlphaAnythingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AlphaAnythingLoRALoader,
            AlphaAnythingElementPrompts,
            AlphaAnythingVaeDecode,
        ]


async def comfy_entrypoint() -> AlphaAnythingExtension:
    return AlphaAnythingExtension()
