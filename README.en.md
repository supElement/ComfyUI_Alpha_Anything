<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_Alpha_Anything

Native ComfyUI implementation of AlphaAnything: does not rely on diffusers, LoRA loads via ComfyUI's native loader, all model loading, text encoding, sampling, and VAE encoding use built-in nodes; VAE decoding reuses the native decoder, with only an alpha branch attached to its features. The extension only adds two runtime nodes (LoRA loading, RGBA decoding) and an optional Element Prompts prompt node. Connect upstream to a Generate Text (needs a text encoder that supports image input) output layer plan that can read images, and you can extract a set of RGBA layers layer by layer in one queue: each layer gets a full sampling, the foreground layer has true transparency, and the background layer is an opaque full image.

</div>

<img width="2240" height="766" alt="image" src="https://github.com/user-attachments/assets/b36ecda2-8058-4e95-b01c-ad707f425161" />

<img width="2241" height="1135" alt="image" src="https://github.com/user-attachments/assets/95d2be5a-cac6-4c10-90db-a8138d93a7d3" />

---

## What Problem It Solves

AlphaAnything trained two things on RGB diffusion models (FLUX.2 Klein / Qwen-Image): transformer LoRA, and an alpha decoding branch.

- RGB generation proceeds normally, alpha **does not enter latent**.
- During decoding, the alpha branch bypasses the intermediate features of the RGB VAE decoder (conv_in / mid / up stages), predicts alpha, and then maps it to 0–1.

This extension hands over all the parts that ComfyUI already has to native nodes, only adding LoRA loading and VAE decoding, and adds an auxiliary node AlphaAnything Element Prompts. Through the auxiliary node, you can generate multiple layers with one click, and the layer accuracy depends on the VLM model's capability (of course, you can also input manually).

- **LoRA Loading**: Takes official LoRA weights, converts them according to the alpha/rank configured in the adapter, and loads them into MODEL, passing down the alpha branch weights along with the adapter.
- **Decoding**: Hooks into the decoder features during the native `vae.decode` execution process to drive the alpha branch, outputting 4-channel RGBA.

---

## Supported Models and Adapters

| Base Model | Task | Adapter Directory | Required VAE |
|---|---|---|---|
| FLUX.2 Klein Base 9B | Text-to-RGBA | `flux2_klein_base_9b_t2rgba` | FLUX.2 VAE（128 channel latent，2D decoder） |
| FLUX.2 Klein Base 9B | Image-to-RGBA | `flux2_klein_base_9b_i2rgba` | Same as above |
| Qwen-Image-2512 | Text-to-RGBA | `qwen_image_2512_t2rgba` | Qwen-Image VAE（Wan 2.1，16 channel） |
| Qwen-Image-Edit-2511 | Image-to-RGBA | `qwen_image_edit_2511_i2rgba` | Same as above |

> Adapters are released by [WLuLi/AlphaAnything](https://huggingface.co/WLuLi/AlphaAnything), each about 400–470MB.
>
> The two FLUX.2 adapters were trained on Klein Base and can also be used directly for 4-step distillation version `FLUX.2-klein-9B`（most cases, the official instructions are usable）.

---

## 📦 Installation

### Manager Installation

Search for **ComfyUI_Alpha_Anything** in the ComfyUI Manager and then Install.

### Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_Alpha_Anything.git
```

### Model Directory

Adapters are placed uniformly in:

```text
 ComfyUI/models/alphaanything/
   flux2_klein_base_9b_t2rgba/
     adapter_config.json
     pytorch_lora_weights.safetensors
     alpha_decoder.safetensors
   qwen_image_2512_t2rgba/
     ...
```

- If the directory does not exist, it will be created automatically. The first time the adapter is executed, it will be downloaded automatically; the download first goes `huggingface.co`, and if it fails, it goes `hf-mirror.com`.
- You can also manually download the corresponding subdirectory from [Hugging Face](https://huggingface.co/WLuLi/AlphaAnything/tree/main) and place it in the structure above.

---

## Node Overview

Nodes are located in `alphaanything` category, search `AlphaAnything` to find them.

| Node | Function | Input | Output |
|---|---|---|---|
| **AlphaAnything LoRA Loader** | Loads official LoRA and alpha branch weights, attaching them to MODEL | `adapter`、`strength`、`model` | `model`、`alpha` |
| **AlphaAnything Element Prompts** | Takes the layer plan from the VLM（or manually entered list）and fits it into a template to generate a prompt list | `vlm`、`use_vlm`、`elements`、`template`、`background` | `prompts`（list）、`with_alpha`（list） |
| **AlphaAnything VAE Decode (RGBA)** | Uses native VAE decoding while outputting alpha and synthesizing RGBA | `samples`、`vae`、`alpha`、`with_alpha` | `IMAGE`（4 channels） |

- `AlphaAnything LoRA Loader` connects between the model loading node and KSampler.
- `AlphaAnything Element Prompts` node connects to VLM output, `use_vlm` turns on to automatically split `<ELEMENTS>` / `<BACKGROUND>`; when off, use `elements` / `background` to manually enter content, keeping port connections.
- In VLM mode, after each execution, the split results are automatically refilled to `elements` / `background` input boxes for easy viewing and editing (to make edits take effect, you need to turn off the use_vlm switch); the content used in this queue still uses the refilled content, and cache reruns will also refill.
- `template` element rows fit into the template（if `{element}` is present, they replace it; otherwise, they append to the end），the background layer command is used as is, without fitting the template.
- `AlphaAnything VAE Decode (RGBA)` replaces the normal VAE Decode; `alpha` port connects to LoRA Loader's `alpha` output.
- `with_alpha` is layer tagging: Element Prompts' `with_alpha` output matches `prompts` in length（background layer off），when connected to the decoding node, the background layer automatically skips the alpha branch and outputs an opaque full image; the two ports must be on the same chain to ensure layer alignment. If this port is not connected, the decoding node calculates alpha for all layers.
## Usage

### Text-to-RGBA

- Load the component according to the model's native workflow: Connect `AlphaAnything LoRA Loader`, `adapter` and select **t2rgba**.
- Set the sampling parameters according to the official reference: Klein base 50 steps / CFG 4.0; 4-step distillation version 4 steps / CFG 1.0; Qwen-Image 50 steps / CFG 4.0.
- Connect `AlphaAnything VAE Decode (RGBA)` after sampling.

### Image-to-RGBA

Integrate this extension into the corresponding model's native editing workflow:

- Replace `AlphaAnything LoRA Loader` with the **i2rgba** adapter. The output still goes through `AlphaAnything VAE Decode (RGBA)`.

---

## Prompts

There is no mandatory template from the official side. Three types of writing exist simultaneously in the web demo, choose one based on the target.

### Text-to-RGBA

The generation examples on the official website primarily focus on **detailed descriptions**: main structure, material, transparency representation (glass, fabric, feathers, paper art), light and shadows. It does not necessarily add a fixed set phrase. For example (Qwen-Image):

> A glasswing butterfly opens its nearly transparent wings, edged in chestnut and narrow cream bands. Fine dark veins cross the clear membranes, while a slim opaque body, curled legs, and delicate antennae remain sharply defined. Natural-history macro photography with faint blue reflections on the wings.

README / CLI examples often use a shorter "description + `isolated on a transparent background.`":

> A delicate glass butterfly with translucent wings, isolated on a transparent background.


- Describe one subject at a time (the output will only have one layer of alpha, multiple targets will be merged into the same layer).
- For translucent materials, it is more effective to describe them clearly in the prompt (`translucent`, `clear glass`, `fine mesh`) than to abstractly emphasize "transparency".
- Put unwanted backgrounds, trash in negative prompts.

### Image-to-RGBA

The i2rgba examples on the web use "fixed introduction + target description":

> Extract the object described by the text, preserving fine details and transparency: A clear long-stemmed glass holds a coral-red cocktail with large ice cubes and an orange slice. Bright reflections run down its rounded bowl and slender stem to the oval glass foot.

Layered disassembly examples use free instructions, clearly specifying what to take, preserve original position and transparency, and make the rest transparent:

> Extract only the tall clear stemmed cocktail glass on the right, including its complete rounded bowl, thin rim, coral-red drink, ice, orange garnish inside the bowl, slender stem and oval glass foot. Preserve true natural partial transparency through the empty upper glass, glass walls, clear stem and base, with the original reflections and refraction. Everything outside the requested element must be transparent. Do not recenter, resize, crop, restyle or add objects.

Giving a detailed generative description is also feasible (the layer writing in the official combined examples, used for image-to-RGBA extraction test, is consistent with official results):

> A broad luxurious champagne silk fabric drape lying almost horizontally on an unseen tabletop, with one softly raised folded crest at the right and several smooth gently curved folds trailing toward the left. Heavy fluid satin with muted honey-gold highlights, deep subtle champagne shadows, exceptionally fine woven texture and realistic weighted cloth. A single complete restrained connected fabric shape, low and wide. The entire complete subject is inside generous empty margins, isolated on a genuinely transparent background. Authentic fine material detail, restrained natural colors, soft directional illumination from upper left.

When layering, don't miss the background layer: the background layer is not "the rest transparent", but to keep the entire image, remove the foreground one by one, and output an opaque layer:

> Extract the complete photographic BACKGROUND, keeping the original dark studio wall, the gray stone countertop and the plants behind the glass. Remove only the cocktail glass and drink, both mint leaves, chrome shaker, all words and icons. Reconstruct the wall, foliage and counter only where those foreground elements were removed. Return the entire original rectangular background fully opaque, preserving the original lighting, blur and framing.


- Fill the background layer instruction in `AlphaAnything Element Prompts`'s `background` (original usage, placed first), `elements` only put foreground elements.
- The `with_alpha` for the background layer is automatically set to off by Element Prompts. Connect this output to the decoding node, no need to split branches; the background layer still runs according to independent sampling, will not appear occasional hollowing.
## Parameter Description

| Parameter | Description |
|---|---|
| `adapter` | Official 4 adapters + local custom directory (must contain `adapter_config.json`). Text-to-image choose t2rgba, image-to-image choose i2rgba. |
| `strength` | Default 1.0 = Official strength. Internally loads LoRA according to `strength × alpha / rank`. |

---

## Known Limitations

1. **Only supports two types of VAE decoders**: FLUX.2's 2D Decoder and Qwen-Image (Wan 2.1) Decoder3d.
2. **Does not support tiled decode**: Hit the tiled path will report a clear error. Insufficient GPU memory, please reduce resolution, do not enable tiled.

---

## Acknowledgements and License

- Method, LoRA, and alpha branch: **AlphaAnything** (WLuLi et al.), code Apache-2.0, adapter weights see [Hugging Face](https://huggingface.co/WLuLi/AlphaAnything).
- Upstream base model: FLUX.2 Klein for non-commercial license, Qwen-Image for Apache-2.0; please confirm the corresponding license before use and distribution.
- Decoder implementation is completely based on ComfyUI's native VAE decoder, no modification to ComfyUI core code.

---

## About Sponsorship

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/supElement)

**⚠️ Note:**
Whether you choose to sponsor or not, it will have no impact on the features or user experience provided by my plugin. All content will remain free and open-source.
**Please only consider sponsoring if you genuinely find these tools helpful, or if you have gained benefits from using them.**