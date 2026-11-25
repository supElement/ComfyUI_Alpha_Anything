from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.modules.diffusionmodules.model
import comfy.ldm.wan.vae


class AlphaTopology(NamedTuple):
    conv_in_channels: int
    mid_rgb_channels: tuple[int, ...]
    up_rgb_channels: tuple[tuple[int, ...], ...]
    has_upsample: tuple[bool, ...]


def _group_count(channels: int) -> int:
    for groups in range(min(int(channels), 32), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _out_channels(module: nn.Module) -> int:
    for name in ("out_channels", "out_dim"):
        value = getattr(module, name, None)
        if value is not None:
            return int(value)
    raise TypeError(f"Cannot infer output channels for decoder block {type(module)!r}.")


def _spatial_resize_nearest(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, channels, frames, height, width = value.shape
    if (height, width) == tuple(size):
        return value
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat = F.interpolate(flat, size=size, mode="nearest-exact")
    return flat.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)


def ea_score_to_alpha(score: torch.Tensor) -> torch.Tensor:
    coordinate = ((score.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    return torch.sin(coordinate * (math.pi / 2.0)).square()


class RGBToAlphaLoCon(nn.Module):
    """Low-rank spatial projection reading a frozen RGB decoder feature."""

    def __init__(self, in_channels: int, out_channels: int, rank: int, *, alpha: float) -> None:
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Conv3d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv3d(rank, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)

    def forward(self, rgb_feature: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(rgb_feature)) * self.scaling


class AlphaResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return value + residual


class AlphaInit(nn.Module):
    """Initialize the alpha branch state from the RGB decoder's conv_in feature."""

    def __init__(self, rgb_channels: int, alpha_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(rgb_channels, alpha_channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(alpha_channels), alpha_channels)
        self.spatial = nn.Conv3d(alpha_channels, alpha_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, rgb_feature: torch.Tensor) -> torch.Tensor:
        return self.spatial(F.silu(self.norm(self.proj(rgb_feature))))


class RGBToAlphaCrossAttention(nn.Module):
    """One-way bottleneck cross-attention: alpha queries frozen RGB keys/values."""

    def __init__(self, alpha_channels: int, rgb_channels: int, *, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(alpha_channels) // self.heads
        self.alpha_norm = nn.GroupNorm(_group_count(alpha_channels), alpha_channels)
        self.rgb_norm = nn.GroupNorm(_group_count(rgb_channels), rgb_channels)
        self.to_q = nn.Conv2d(alpha_channels, alpha_channels, kernel_size=1)
        self.to_k = nn.Conv2d(rgb_channels, alpha_channels, kernel_size=1)
        self.to_v = nn.Conv2d(rgb_channels, alpha_channels, kernel_size=1)
        self.to_out = nn.Conv2d(alpha_channels, alpha_channels, kernel_size=1)

    def _tokens(self, value: torch.Tensor) -> torch.Tensor:
        batch_frames, _, height, width = value.shape
        value = value.flatten(2).transpose(1, 2).contiguous()
        return value.reshape(batch_frames, height * width, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        batch, alpha_channels, frames, height, width = alpha.shape
        alpha_2d = alpha.permute(0, 2, 1, 3, 4).reshape(batch * frames, alpha_channels, height, width)
        rgb_2d = rgb.permute(0, 2, 1, 3, 4).reshape(batch * frames, rgb.shape[1], height, width)
        query = self._tokens(self.to_q(self.alpha_norm(alpha_2d)))
        norm_rgb = self.rgb_norm(rgb_2d)
        key = self._tokens(self.to_k(norm_rgb))
        value = self._tokens(self.to_v(norm_rgb))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch * frames, height * width, alpha_channels)
        attended = self.to_out(attended.transpose(1, 2).reshape(batch * frames, alpha_channels, height, width))
        return alpha + attended.reshape(batch, frames, alpha_channels, height, width).permute(0, 2, 1, 3, 4)


class LearnedAlphaUpsample(nn.Module):
    """Nearest spatial resize followed by a 3x3 spatial convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.conv(_spatial_resize_nearest(value, size))


class AlphaDecoderStage(nn.Module):
    """Upsampling stage of the alpha branch reading RGB ResBlock features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rgb_feature_channels: tuple[int, ...],
        *,
        injection_rank: int,
        injection_alpha: float,
        res_blocks: int,
        learned_upsample: bool,
        injection_gate_init: float,
    ) -> None:
        super().__init__()
        if int(in_channels) == int(out_channels):
            self.transition: nn.Module = nn.Identity()
        else:
            self.transition = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.injections = nn.ModuleList([
            RGBToAlphaLoCon(rgb_c, out_channels, injection_rank, alpha=injection_alpha)
            for rgb_c in rgb_feature_channels
        ])
        self.injection_normalization = 1.0 / math.sqrt(len(self.injections))
        self.injection_gate = nn.Parameter(torch.tensor(float(injection_gate_init), dtype=torch.float32))
        self.blocks = nn.ModuleList([AlphaResBlock(out_channels) for _ in range(res_blocks)])
        self.upsample = LearnedAlphaUpsample(out_channels) if learned_upsample else None

    def begin(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.transition(alpha)

    def inject(self, alpha: torch.Tensor, rgb_feature: torch.Tensor, index: int) -> torch.Tensor:
        contrib = self.injections[index](rgb_feature)
        scale = self.injection_gate.to(dtype=contrib.dtype) * self.injection_normalization
        return alpha + contrib * scale

    def finish(self, alpha: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            alpha = block(alpha)
        return alpha

    def forward_upsample(self, alpha: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if self.upsample is None:
            return alpha
        return self.upsample(alpha, size)


class AlphaMidStage(nn.Module):
    """Bottleneck alpha stage with one RGB -> alpha cross-attention block."""

    def __init__(
        self,
        channels: int,
        rgb_feature_channels: tuple[int, ...],
        *,
        injection_rank: int,
        injection_alpha: float,
        res_blocks: int,
        attention_heads: int,
        injection_gate_init: float,
    ) -> None:
        super().__init__()
        self.injections = nn.ModuleList([
            RGBToAlphaLoCon(rgb_c, channels, injection_rank, alpha=injection_alpha)
            for rgb_c in rgb_feature_channels
        ])
        self.injection_normalization = 1.0 / math.sqrt(len(self.injections))
        self.injection_gate = nn.Parameter(torch.tensor(float(injection_gate_init), dtype=torch.float32))
        self.blocks = nn.ModuleList([AlphaResBlock(channels) for _ in range(res_blocks)])
        self.cross_attention = RGBToAlphaCrossAttention(
            channels, rgb_feature_channels[-1], heads=attention_heads,
        )

    def inject(self, alpha: torch.Tensor, rgb_feature: torch.Tensor, index: int) -> torch.Tensor:
        contrib = self.injections[index](rgb_feature)
        scale = self.injection_gate.to(dtype=contrib.dtype) * self.injection_normalization
        return alpha + contrib * scale

    def finish(self, alpha: torch.Tensor, final_rgb_feature: torch.Tensor) -> torch.Tensor:
        alpha = self.blocks[0](alpha)
        alpha = self.cross_attention(alpha, final_rgb_feature)
        for block in self.blocks[1:]:
            alpha = block(alpha)
        return alpha


class EAScoreHead(nn.Module):
    """Alpha head predicting the raw score for the Endpoint-Aware alpha mapping."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.blocks = nn.ModuleList([AlphaResBlock(hidden_channels) for _ in range(2)])
        self.output = nn.Conv3d(hidden_channels, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha = self.input(alpha)
        for block in self.blocks:
            alpha = block(alpha)
        return self.output(alpha)


class BlockTriangularAlphaDecoder(nn.Module):
    """Hierarchical alpha branch with strictly one-way RGB feature sharing."""

    def __init__(self, topology: AlphaTopology, config: dict) -> None:
        super().__init__()
        stage_channels = [int(v) for v in config.get("stage_channels", [256, 256, 192, 128, 96])]
        stage_gates = [float(v) for v in config.get("injection_stage_gates", [1.0, 0.75, 0.5, 0.25, 0.25])]
        injection_rank = int(config.get("injection_rank", 64))
        injection_alpha = float(config.get("injection_alpha", injection_rank))
        res_blocks = int(config.get("res_blocks_per_stage", 2))
        attention_heads = int(config.get("attention_heads", 8))
        head_channels = int(config.get("head_channels", 128))

        self.alpha_init = AlphaInit(topology.conv_in_channels, stage_channels[0])
        self.mid_stage = AlphaMidStage(
            stage_channels[0],
            topology.mid_rgb_channels,
            injection_rank=injection_rank,
            injection_alpha=injection_alpha,
            res_blocks=res_blocks,
            attention_heads=attention_heads,
            injection_gate_init=stage_gates[0],
        )
        self.up_stages = nn.ModuleList()
        prev_channels = stage_channels[0]
        for out_channels, rgb_channels, gate_init, learned_upsample in zip(
            stage_channels[1:], topology.up_rgb_channels, stage_gates[1:], topology.has_upsample, strict=True,
        ):
            self.up_stages.append(
                AlphaDecoderStage(
                    prev_channels,
                    out_channels,
                    rgb_channels,
                    injection_rank=injection_rank,
                    injection_alpha=injection_alpha,
                    res_blocks=res_blocks,
                    learned_upsample=learned_upsample,
                    injection_gate_init=gate_init,
                )
            )
            prev_channels = out_channels
        self.ea_score_head = EAScoreHead(stage_channels[-1], head_channels)


def _decoder_layout(decoder: nn.Module):
    """Return (conv_in, mid resnets, up stages in decode order, 2D?) of a native RGB decoder."""
    if isinstance(decoder, comfy.ldm.wan.vae.Decoder3d):
        mid = [layer for layer in decoder.middle if isinstance(layer, comfy.ldm.wan.vae.ResidualBlock)]
        stages = []
        layers = []
        for layer in decoder.upsamples:
            if isinstance(layer, comfy.ldm.wan.vae.Resample):
                stages.append((layers, layer))
                layers = []
            elif isinstance(layer, comfy.ldm.wan.vae.ResidualBlock):
                layers.append(layer)
        stages.append((layers, None))
        return decoder.conv1, mid, stages, False

    if isinstance(decoder, comfy.ldm.modules.diffusionmodules.model.Decoder):
        mid = [decoder.mid.block_1, decoder.mid.block_2]
        stages = [(list(up.block), getattr(up, "upsample", None)) for up in reversed(decoder.up)]
        return decoder.conv_in, mid, stages, True

    raise TypeError("AlphaAnything needs a FLUX.2 2D VAE or a Qwen-Image (Wan 2.1) VAE.")


def decode_rgba(vae, samples: torch.Tensor, adapter: dict, *, with_alpha: bool = True) -> torch.Tensor:
    """Decode latents with the native VAE decoder while the alpha branch reads its features."""
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise RuntimeError("AlphaAnything decodes single images only.")

    if not with_alpha:
        pixels = vae.decode(samples)
        if pixels.ndim == 5:
            pixels = pixels[:, 0]
        opaque = torch.ones((*pixels.shape[:3], 1), dtype=pixels.dtype, device=pixels.device)
        return torch.cat((pixels, opaque), dim=-1)

    conv_in, mid_layers, stages, two_dimensional = _decoder_layout(vae.first_stage_model.decoder)
    topology = AlphaTopology(
        conv_in_channels=_out_channels(conv_in),
        mid_rgb_channels=tuple(_out_channels(module) for module in mid_layers),
        up_rgb_channels=tuple(tuple(_out_channels(module) for module in layers) for layers, _ in stages),
        has_upsample=tuple(upsampler is not None for _, upsampler in stages),
    )
    alpha = BlockTriangularAlphaDecoder(topology, adapter["config"])
    try:
        alpha.load_state_dict(adapter["state"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"AlphaAnything adapter {adapter['backbone']!r} does not match this VAE.") from error
    alpha = alpha.to(device=vae.device, dtype=vae.vae_dtype)

    scores = []
    current = {}

    def feature(output):
        return output.unsqueeze(2) if two_dimensional else output

    def start(output):
        current["alpha"] = alpha.alpha_init(feature(output))

    def make_mid(index):
        final = index == len(mid_layers) - 1

        def action(output):
            rgb_feature = feature(output)
            current["alpha"] = alpha.mid_stage.inject(current["alpha"], rgb_feature, index)
            if final:
                current["alpha"] = alpha.mid_stage.finish(current["alpha"], rgb_feature)

        return action

    def make_up(stage_index, index):
        stage = alpha.up_stages[stage_index]
        final_layer = index == len(stages[stage_index][0]) - 1
        final_stage = stage_index == len(stages) - 1

        def action(output):
            if index == 0:
                current["alpha"] = stage.begin(current["alpha"])
            current["alpha"] = stage.inject(current["alpha"], feature(output), index)
            if final_layer:
                current["alpha"] = stage.finish(current["alpha"])
                if final_stage:
                    scores.append(alpha.ea_score_head(current["alpha"]))

        return action

    def make_upsample(stage_index):
        stage = alpha.up_stages[stage_index]

        def action(output):
            current["alpha"] = stage.forward_upsample(current["alpha"], tuple(output.shape[-2:]))

        return action

    def make_hook(action):
        def hook(module, args, output):
            action(output)

        return hook

    plan = [(conv_in, start)]
    plan += [(module, make_mid(index)) for index, module in enumerate(mid_layers)]
    for stage_index, (layers, upsampler) in enumerate(stages):
        plan += [(module, make_up(stage_index, index)) for index, module in enumerate(layers)]
        if upsampler is not None:
            plan.append((upsampler, make_upsample(stage_index)))

    handles = [module.register_forward_hook(make_hook(action)) for module, action in plan]
    try:
        pixels = vae.decode(samples)
    finally:
        for handle in handles:
            handle.remove()

    score = torch.cat(scores, dim=0)[:, :, 0]
    if pixels.ndim == 5:
        pixels = pixels[:, 0]
    if score.shape[0] != pixels.shape[0] or score.shape[-2:] != pixels.shape[1:3]:
        raise RuntimeError("AlphaAnything cannot run a tiled VAE decode; free VRAM or lower the resolution.")

    alpha_pixels = ea_score_to_alpha(score).movedim(1, -1).to(device=pixels.device, dtype=pixels.dtype)
    return torch.cat((pixels, alpha_pixels), dim=-1)
