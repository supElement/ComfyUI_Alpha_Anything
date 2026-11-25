<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_Alpha_Anything

AlphaAnything 的原生 ComfyUI 实现：不依赖 diffusers，LoRA 走 ComfyUI 原生加载器，模型加载、文本编码、采样和 VAE 编码全部使用自带节点；VAE 解码复用原生 decoder，只在其特征上挂一条 alpha 分支。扩展只新增两个运行节点（LoRA 加载、RGBA 解码）和一个可选的 Element Prompts 提示词节点。上游接 vlm 节点输出图层计划，即可在一次排队中逐层提取出一组 RGBA 图层：每层一次完整采样，前景层带真实半透明，背景层为不透明整图。

</div>

<img width="2240" height="766" alt="image" src="https://github.com/user-attachments/assets/b36ecda2-8058-4e95-b01c-ad707f425161" />

<img width="2241" height="1135" alt="image" src="https://github.com/user-attachments/assets/95d2be5a-cac6-4c10-90db-a8138d93a7d3" />

---

## 它解决什么问题

AlphaAnything 在 RGB 扩散模型（FLUX.2 Klein / Qwen-Image）上训练了两样东西：transformer LoRA，以及一条 alpha 解码分支。

- RGB 照常生成，alpha **不进入 latent**。
- 解码时，alpha 分支旁路读取 RGB VAE decoder 的中间特征（conv_in / mid / 各级 up stage），预测 alpha，再映射到 0–1。

本扩展把 ComfyUI 已经具备的部分全部交还原生节点，只补 lora 加载和 vae 解码，另外增加了辅助节点 AlphaAnything Element Prompts，通过辅助节点可一键生成多个分层，分层准确度取决于 vlm 模型能力（当然，你也可以手动输入）。

- **LoRA 加载**：把官方 LoRA 权重按 adapter 配置的 alpha/rank 换算后打进 MODEL，并把 alpha 分支权重随适配器往下传。
- **解码**：在原生 `vae.decode` 执行过程中 hook decoder 特征驱动 alpha 分支，输出 4 通道 RGBA。

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

---

## 📦 安装

### Manager 安装

在 ComfyUI Manager 中搜索 **ComfyUI_Alpha_Anything**，然后 Install。

### 手动安装

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_Alpha_Anything.git
```

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

- 目录不存在会自动创建。首次执行适配器时会自动下载；下载先走 `huggingface.co`，失败再走 `hf-mirror.com`。
- 也可以手动从 [Hugging Face](https://huggingface.co/WLuLi/AlphaAnything/tree/main) 下载对应子目录，按上面的结构放好即可。

---

## 节点一览

节点位于 `alphaanything` 分类，搜索 `AlphaAnything` 即可。

| 节点 | 作用 | 输入 | 输出 |
|---|---|---|---|
| **AlphaAnything LoRA Loader** | 加载官方 LoRA 与 alpha 分支权重，挂到 MODEL 上 | `adapter`、`strength`、`model` | `model`、`alpha` |
| **AlphaAnything Element Prompts** | 把 VLM 的图层计划（或手填清单）套进模板，生成 prompt 列表 | `vlm`、`use_vlm`、`elements`、`template`、`background` | `prompts`（list）、`with_alpha`（list） |
| **AlphaAnything VAE Decode (RGBA)** | 用原生 VAE 解码，同时输出 alpha 并合成 RGBA | `samples`、`vae`、`alpha`、`with_alpha` | `IMAGE`（4 通道） |

- `AlphaAnything LoRA Loader` 接在模型加载节点与 KSampler 之间。
- `AlphaAnything Element Prompts` 节点接 VLM 输出，`use_vlm` 打开时按 `<ELEMENTS>` / `<BACKGROUND>` 自动拆分；关闭时只用 `elements` / `background` 手填内容，端口连接保留。
- VLM 模式下每次执行完，拆分结果会自动回填到 `elements` / `background` 输入框，方便查看和再编辑（为了使编辑生效需要关闭 use_vlm 开关）；本次排队仍用回填前的内容，缓存重跑也会回填。
- `template` 元素行会套进模板（含 `{element}` 则替换，否则追加在后面），背景层指令原样使用、不套模板。
- `AlphaAnything VAE Decode (RGBA)` 替换普通的 VAE Decode；`alpha` 端口接 LoRA Loader 的 `alpha` 输出。
- `with_alpha` 是逐层标记：Element Prompts 的 `with_alpha` 输出与 `prompts` 等长（背景层为关），接到解码节点后，背景层自动跳过 alpha 分支、输出不透明整图；两个端口要走同一条链路，保证逐层对齐。不接这个端口时，解码节点对所有层都计算 alpha。

---

## 使用

### 文生 RGBA

- 按模型原生工作流加载组件：接入 `AlphaAnything LoRA Loader`，`adapter` 选 **t2rgba**。
- 采样参数按官方参考设置：Klein base 50 步 / CFG 4.0；4-step 蒸馏版 4 步 / CFG 1.0；Qwen-Image 50 步 / CFG 4.0。
- 采样后接 `AlphaAnything VAE Decode (RGBA)`。

### 图生 RGBA

在对应模型的原生编辑工作流基础上接入本扩展：

- `AlphaAnything LoRA Loader` 换成 **i2rgba** 适配器。 输出仍走 `AlphaAnything VAE Decode (RGBA)`。

---

## 提示词

官方没有强制模板，网页 demo 里就同时存在三种写法，按目标选一种即可。

### 文生（Text-to-RGBA）

官方网页的生成示例以**详细描述**为主：主体结构、材质、透明度表现（玻璃、织物、羽毛、纸艺）、光线与影调，不固定加某句套话。例如 (Qwen-Image) ：

> A glasswing butterfly opens its nearly transparent wings, edged in chestnut and narrow cream bands. Fine dark veins cross the clear membranes, while a slim opaque body, curled legs, and delicate antennae remain sharply defined. Natural-history macro photography with faint blue reflections on the wings.

README / CLI 示例常用更短的「描述 + `isolated on a transparent background.`」：

> A delicate glass butterfly with translucent wings, isolated on a transparent background.


- 一次描述一个主体（输出只有一层 alpha，多目标会合并到同一图层）。
- 半透明材质直接在描述里写清楚（`translucent`、`clear glass`、`fine mesh`）比抽象强调“透明”更有效。
- 不想要的背景、杂物放负向提示词。

### 图生（Image-to-RGBA）

网页 i2rgba 示例用「固定引言 + 目标描述」：

> Extract the object described by the text, preserving fine details and transparency: A clear long-stemmed glass holds a coral-red cocktail with large ice cubes and an orange slice. Bright reflections run down its rounded bowl and slender stem to the oval glass foot.

分层拆解示例则用自由指令，明确只取什么、保留原位置与透明度、其余透明：

> Extract only the tall clear stemmed cocktail glass on the right, including its complete rounded bowl, thin rim, coral-red drink, ice, orange garnish inside the bowl, slender stem and oval glass foot. Preserve true natural partial transparency through the empty upper glass, glass walls, clear stem and base, with the original reflections and refraction. Everything outside the requested element must be transparent. Do not recenter, resize, crop, restyle or add objects.

直接给详细的生成式描述同样可行（官方组合示例中的图层写法，用于图生提取实测与官方结果一致）：

> A broad luxurious champagne silk fabric drape lying almost horizontally on an unseen tabletop, with one softly raised folded crest at the right and several smooth gently curved folds trailing toward the left. Heavy fluid satin with muted honey-gold highlights, deep subtle champagne shadows, exceptionally fine woven texture and realistic weighted cloth. A single complete restrained connected fabric shape, low and wide. The entire complete subject is inside generous empty margins, isolated on a genuinely transparent background. Authentic fine material detail, restrained natural colors, soft directional illumination from upper left.

分层时别漏背景层：背景层不是「其余透明」，而是保留整幅画面、逐个去掉前景，输出不透明图层：

> Extract the complete photographic BACKGROUND, keeping the original dark studio wall, the gray stone countertop and the plants behind the glass. Remove only the cocktail glass and drink, both mint leaves, chrome shaker, all words and icons. Reconstruct the wall, foliage and counter only where those foreground elements were removed. Return the entire original rectangular background fully opaque, preserving the original lighting, blur and framing.


- 背景层指令填到 `AlphaAnything Element Prompts` 的 `background`（原样使用、排在首位），`elements` 只放前景元素。
- 背景层的 `with_alpha` 由 Element Prompts 自动置为关，把该输出接到解码节点即可，无需拆分支；背景层仍按独立采样跑，不会出现偶发镂空。

---

## 参数说明

| 参数 | 说明 |
|---|---|
| `adapter` | 官方 4 个适配器 + 本地自定义目录（需含 `adapter_config.json`）。文生选 t2rgba，图生选 i2rgba。 |
| `strength` | 默认 1.0 = 官方强度。内部按 `strength × alpha / rank` 加载 LoRA。 |

---

## 已知限制

1. **仅支持两类 VAE 解码器**：FLUX.2 的 2D Decoder 和 Qwen-Image（Wan 2.1）的 Decoder3d。
2. **不支持 tiled decode**：命中 tiled 路径会报明确错误。显存不足请降低分辨率，不要开 tiled。

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
