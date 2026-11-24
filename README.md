<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_Alpha_Anything

AlphaAnything 的纯原生 ComfyUI 实现：不依赖 diffusers，只新增一个 LoRA 加载节点和一个 RGBA 解码节点，模型加载、文本编码、采样、VAE 编解码全部使用 ComfyUI 自带节点。

</div>

---

## 它解决什么问题

AlphaAnything 在 RGB 扩散模型（FLUX.2 Klein / Qwen-Image）上训练了两样东西：transformer LoRA，以及一条 alpha 解码分支。

- RGB 照常生成，alpha **不进入 latent**。
- 解码时，alpha 分支旁路读取 RGB VAE decoder 的中间特征（conv_in / mid / 各级 up stage），预测 alpha，再映射到 0–1。
- 因此官方实现必须改动 diffusers 的解码流程，无法直接套用原生 ComfyUI 工作流。

本扩展把 ComfyUI 已经具备的部分全部交还原生节点，只补两件事：

- **LoRA 加载**：把官方 LoRA 权重按 adapter 配置的 alpha/rank 换算后打进 MODEL，并把 alpha 分支权重随适配器往下传。
- **解码**：在原生 `vae.decode` 执行过程中 hook decoder 特征驱动 alpha 分支，输出 4 通道 RGBA。

✨ **RGB 逐像素一致**：RGB 通道与直接调用原生 `vae.decode` 结果完全相同（差值为 0），alpha 由旁路分支生成。

✨ **强度自动换算**：`strength = 1.0` 就是官方强度；adapter 配置里的 `alpha / rank` 会自动乘进去。

✨ **适配器自动下载**：只读 `ComfyUI/models/alphaanything/`，目录不存在自动创建；缺权重文件时首次执行自动下载，Hugging Face 官方源优先，失败再走 hf-mirror 镜像。

---

## 支持的模型与适配器

| 基础模型 | 任务 | 适配器目录 | 需要的 VAE |
|---|---|---|---|
| FLUX.2 Klein Base 9B | 文生 RGBA | `flux2_klein_base_9b_t2rgba` | FLUX.2 VAE（128 通道 latent，2D decoder） |
| FLUX.2 Klein Base 9B | 图生 RGBA | `flux2_klein_base_9b_i2rgba` | 同上 |
| Qwen-Image-2512 | 文生 RGBA | `qwen_image_2512_t2rgba` | Qwen-Image VAE（Wan 2.1，16 通道） |
| Qwen-Image-Edit-2511 | 图生 RGBA | `qwen_image_edit_2511_i2rgba` | 同上 |

> 适配器由 [WLuLi/AlphaAnything](https://huggingface.co/WLuLi/AlphaAnything) 发布，每个约 400–470MB。
>
> 两个 FLUX.2 适配器训练于 Klein Base，也可直接用于 4-step 蒸馏版 `FLUX.2-klein-9B`（官方说明多数情况可用）。
>
> 只支持上述两类 VAE 解码器；Qwen-Image 2.1 的 64 通道 VAE、Wan 2.2 等其他解码器会直接报错。

---

## 📦 安装

### 手动安装

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_Alpha_Anything.git
```

重启 ComfyUI。无额外依赖：下载使用 Python 标准库，LoRA 加载走 ComfyUI 原生 LoRA 加载器。

> 需要较新的 ComfyUI（使用 V3 节点 API 与 `io.Custom` 端口）。

### 模型目录

适配器统一放在：

```text
ComfyUI/models/alphaanything/
  flux2_klein_base_9b_t2rgba/
    adapter_config.json
    pytorch_lora_weights.safetensors
    alpha_decoder.safetensors
  qwen_image_2512_t2rgba/
    ...
```

- 目录不存在会自动创建。
- 三个文件缺任意一个，首次执行该适配器时会自动下载；下载先走 `huggingface.co`，失败再走 `hf-mirror.com`。
- 也可以手动从 [Hugging Face](https://huggingface.co/WLuLi/AlphaAnything/tree/main) 下载对应子目录，按上面的结构放好即可。
- 适配器下拉框默认列出官方 4 个；本地自建的适配器目录只要包含 `adapter_config.json` 也会出现在列表里。

---

## 节点一览

节点位于 `alphaanything` 分类，搜索 `AlphaAnything` 即可。

| 节点 | 作用 | 输入 | 输出 |
|---|---|---|---|
| **AlphaAnything LoRA Loader** | 加载官方 LoRA 与 alpha 分支权重，挂到 MODEL 上 | `adapter`、`strength`、`model` | `model`、`alpha` |
| **AlphaAnything Element Prompts** | 把 VLM 的图层计划（或手填清单）套进模板，生成 prompt 列表 | `vlm`、`use_vlm`、`elements`、`template`、`background` | `prompts`（list） |
| **AlphaAnything VAE Decode (RGBA)** | 用原生 VAE 解码，同时输出 alpha 并合成 RGBA | `samples`、`vae`、`alpha` | `IMAGE`（4 通道） |

- `AlphaAnything LoRA Loader` 接在模型加载节点与 KSampler 之间，文本编码不需要经过它。
- `AlphaAnything Element Prompts` 的 `prompts` 接原生 CLIP 编码节点；list 由 ComfyUI 原生逐项展开，每个元素跑一次完整采样。
- `vlm` 接 VLM 输出，`use_vlm` 打开时按 `<ELEMENTS>` / `<BACKGROUND>` 自动拆分；关闭时只用 `elements` / `background` 手填内容，端口连接保留。
- `AlphaAnything VAE Decode (RGBA)` 替换普通的 VAE Decode；`alpha` 端口接 LoRA Loader 的 `alpha` 输出。

---

## 使用

### 文生 RGBA

1. 按模型原生工作流加载组件：
   - FLUX.2：`UNETLoader`（Load Diffusion Model）选 FLUX.2 Klein base，`CLIPLoader`（Load CLIP）类型选 `flux2`，`VAELoader`（Load VAE）选 FLUX.2 VAE。
   - Qwen-Image：`UNETLoader`（Load Diffusion Model）选 Qwen-Image-2512，`CLIPLoader`（Load CLIP）类型选 `qwen_image`，`VAELoader`（Load VAE）选 Qwen-Image（Wan 2.1）VAE。
2. 空 latent：FLUX.2 用 `Empty Flux 2 Latent`（1024×1024）；Qwen 用 16 通道的 `EmptySD3LatentImage`。
3. 文本编码用原生 CLIP 节点；负向提示词留空或一个空格即可。
4. 接入 `AlphaAnything LoRA Loader`，`adapter` 选 **t2rgba**，`strength` 保持 1.0。
5. 采样参数按官方参考设置：Klein base 50 步 / CFG 4.0；4-step 蒸馏版 4 步 / CFG 1.0；Qwen-Image 50 步 / CFG 4.0。
6. 采样后接 `AlphaAnything VAE Decode (RGBA)`，再保存（`SaveImage` 可直接存 RGBA PNG）。

### 图生 RGBA

在对应模型的原生编辑工作流基础上接入本扩展：

- **FLUX.2**：`LoadImage` → `VAEEncode`（FLUX.2 VAE）→ `Set Reference Latent` 挂到 conditioning；采样空 latent 的宽高与源图一致。
- **Qwen-Image-Edit**：`Text Encode Qwen Image Edit`（`image` 接源图，`vae` 接 Qwen VAE，`prompt` 写提取指令或详细描述，见「提示词」）→ 直接出 conditioning；采样空 latent 宽高与源图一致。
- `AlphaAnything LoRA Loader` 换成 **i2rgba** 适配器，强度仍保持 1.0。
- 采样参数同上；输出仍走 `AlphaAnything VAE Decode (RGBA)`。

> 官方脚本会把源图等比缩放到接近 1024² 再送入模型。在 ComfyUI 里请自行把源图缩放到约 1MP、并让采样 latent 宽高与参考图一致。

---

## 提示词

官方没有强制模板，网页 demo 里就同时存在三种写法，按目标选一种即可。

### 文生（Text-to-RGBA）

官方网页的生成示例以**详细描述**为主：主体结构、材质、透明度表现（玻璃、织物、羽毛、纸艺）、光线与影调，不固定加某句套话。例如 (Qwen-Image) ：

```text
A glasswing butterfly opens its nearly transparent wings, edged in chestnut and narrow cream bands. Fine dark veins cross the clear membranes, while a slim opaque body, curled legs, and delicate antennae remain sharply defined. Natural-history macro photography with faint blue reflections on the wings.
```

README / CLI 示例常用更短的「描述 + `isolated on a transparent background.`」：

```text
A delicate glass butterfly with translucent wings, isolated on a transparent background.
```

- 一次描述一个主体（输出只有一层 alpha，多目标会合并到同一图层）。
- 半透明材质直接在描述里写清楚（`translucent`、`clear glass`、`fine mesh`）比抽象强调“透明”更有效。
- 不想要的背景、杂物放负向提示词。

### 图生（Image-to-RGBA）

网页 i2rgba 示例用「固定引言 + 目标描述」：

```text
Extract the object described by the text, preserving fine details and transparency: A clear long-stemmed glass holds a coral-red cocktail with large ice cubes and an orange slice. Bright reflections run down its rounded bowl and slender stem to the oval glass foot.
```

分层拆解示例则用自由指令，明确只取什么、保留原位置与透明度、其余透明：

```text
Extract only the tall clear stemmed cocktail glass on the right, including its complete rounded bowl, thin rim, coral-red drink, ice, orange garnish inside the bowl, slender stem and oval glass foot. Preserve true natural partial transparency through the empty upper glass, glass walls, clear stem and base, with the original reflections and refraction. Everything outside the requested element must be transparent. Do not recenter, resize, crop, restyle or add objects.
```

直接给详细的生成式描述同样可行（官方组合示例中的图层写法，用于图生提取实测与官方结果一致）：

```text
A broad luxurious champagne silk fabric drape lying almost horizontally on an unseen tabletop, with one softly raised folded crest at the right and several smooth gently curved folds trailing toward the left. Heavy fluid satin with muted honey-gold highlights, deep subtle champagne shadows, exceptionally fine woven texture and realistic weighted cloth. A single complete restrained connected fabric shape, low and wide. The entire complete subject is inside generous empty margins, isolated on a genuinely transparent background. Authentic fine material detail, restrained natural colors, soft directional illumination from upper left.
```

分层时别漏背景层：背景层不是「其余透明」，而是保留整幅画面、逐个去掉前景，输出不透明图层：

```text
Extract the complete photographic BACKGROUND, keeping the original dark studio wall, the gray stone countertop and the plants behind the glass. Remove only the cocktail glass and drink, both mint leaves, chrome shaker, all words and icons. Reconstruct the wall, foliage and counter only where those foreground elements were removed. Return the entire original rectangular background fully opaque, preserving the original lighting, blur and framing.
```

- 背景层指令填到 `AlphaAnything Element Prompts` 的 `background`（原样使用、排在首位），`elements` 只放前景元素。
- 位置词按**源图**中的位置描述（on the left / the smaller line 等）。
- 一次提取一个目标；要分层就多次运行。
- 不限于半透明元素，海报里的不透明物体、文字、装饰图形同样可以提取。
- 本扩展不会替你加前缀，文本原样进编码器；想复现官方 CLI 的 `--description` 效果，自己带上那句 `Extract the object described by the text, ...` 即可。


---

## 参数说明

| 参数 | 说明 |
|---|---|
| `adapter` | 官方 4 个适配器 + 本地自定义目录（需含 `adapter_config.json`）。文生选 t2rgba，图生选 i2rgba。 |
| `strength` | 默认 1.0 = 官方强度。内部按 `strength × alpha / rank` 加载 LoRA。 |
| `samples` | 采样 latent，解码时只取 `samples["samples"]`。 |
| `vae` | 必须与适配器家族匹配（FLUX.2 配 FLUX.2 VAE，Qwen 配 Qwen Wan 2.1 VAE）。 |

> `strength` 只缩放 LoRA，不缩放 alpha 分支。两者是配套训练的，随意调低会让 RGB 偏离 alpha 分支的训练分布，建议保持 1.0。

---

## 已知限制

1. **仅支持两类 VAE 解码器**：FLUX.2 的 2D Decoder 和 Qwen-Image（Wan 2.1）的 Decoder3d。其他结构（含 Qwen-Image 2.1 的 64 通道 VAE）会报错，不会静默降级。
2. **只解单帧图像**：latent 时间维大于 1 会报错。
3. **不支持 tiled decode**：命中 tiled 路径会报明确错误。显存不足请降低分辨率，不要开 tiled。
4. **图生不会自动加前缀**：文本原样进原生编码器；官方前缀形式、自由指令、详细描述都可用（见「提示词」）。
5. **输出是 4 通道 IMAGE**：`SaveImage` 可正常保存 RGBA PNG；只接受 3 通道的下游节点可能会报错。
6. **适配器与底模要配套**：FLUX.2 适配器只能用 FLUX.2 VAE，Qwen 适配器只能用 Qwen VAE；不匹配时解码会报 “does not match this VAE”。
7. **4-step 蒸馏底模可用但不保证**：适配器在 Klein base 上训练，官方说明 4-step 版多数情况可用，偶尔不如 base 稳定。
8. **本扩展不提供 i2rgba 的条件节点**：图生必须走模型原生的参考图 / 编辑编码节点，接到普通 img2img 上不会得到正确结果。

---

## 致谢与许可

- 方法、LoRA 与 alpha 分支：**AlphaAnything**（WLuLi et al.），代码 Apache-2.0，适配器权重见 [Hugging Face](https://huggingface.co/WLuLi/AlphaAnything)。
- 上游底模：FLUX.2 Klein 为非商用许可，Qwen-Image 为 Apache-2.0；使用与分发前请自行确认对应许可。
- 解码实现完全基于 ComfyUI 原生 VAE decoder，未修改 ComfyUI 核心代码。

---

## 关于赞助

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/supElement)

**⚠️ 说明：**
无论您是否选择赞助，都不会对我的插件所提供的功能或使用体验产生任何影响。所有内容将始终保持免费且开源。
**请仅在您确实觉得这些工具对您有所帮助，或者通过使用它们获得了收益的情况下，再考虑进行赞助。**
