import copy
import dataclasses
import enum
import math
from functools import partial
from typing import Any, Callable, Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class VitSize(TypedDict):
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float


VIT_SIZES: dict[str, VitSize] = {
    # Standard sizes
    "debug": VitSize(embed_dim=256, depth=2, num_heads=4, mlp_ratio=2),
    "small": VitSize(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4),
    "base": VitSize(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4),
    "large": VitSize(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4),
    "giant": VitSize(embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4),
    # Scale-optimal models
    "so400m": VitSize(embed_dim=1152, depth=27, num_heads=16, mlp_ratio=4304 / 1152),
}


@torch.no_grad()
def sinkhorn_knopp(
    x: Tensor, temperature: float = 0.1, n_iterations: int = 3
) -> Tensor:
    # x: [(B0, B1, ...,) B, K]

    # Like the numerator of a softmax, subtract vmax for stability
    x = x.float() / temperature
    vmax = x.max()
    # torch.distributed.all_reduce(vmax, op=torch.distributed.ReduceOp.MAX)
    x = x - vmax
    x = x.exp()

    dims1 = tuple(i for i in range(x.ndim - 1))
    dims2 = x.ndim - 1
    for _ in range(n_iterations):
        # Each column sums to 1
        sum_B = x.sum(dim=dims1, keepdim=True)
        # torch.distributed.all_reduce(sum_B)
        x /= sum_B
        # Each row sums to 1, like a probability distribution
        x /= x.sum(axis=dims2, keepdims=True)

    return x


def koleo_batched(x: Tensor) -> Tensor:
    # Same as avg(koleo_single(x[b]) for b in range(B))
    B, N, D = x.shape
    x = x.float()
    x = F.normalize(x, p=2.0, dim=2)  # [B, N, D]
    with torch.no_grad():
        dot = torch.bmm(x, x.transpose(1, 2))  # [B, N, B]
        fill = dot.new_full((B, N), -torch.inf)
        dot = dot.diagonal_scatter(fill, dim1=1, dim2=2)
        idx = dot.argmax(dim=2)  # [B, N]
    pair = x.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
    dist = F.pairwise_distance(x, pair, p=2.0)  # [B, N]
    loss = -torch.log(dist + 1e-8).mean()
    return loss


def cross_entropy(inputs, targets):
    """
    dispatches to cross-entropy, but
    considers 3d tensors as [B, S, K]
    instead of [B, K, S]
    """
    if inputs.ndim == 3:
        inputs = inputs.transpose(1, 2)
        targets = targets.transpose(1, 2)
    return F.cross_entropy(inputs, targets, reduction="none")


@dataclasses.dataclass
class LearningRateSchedule:
    start: float
    peak: float
    end: float
    warmup_iterations: int
    scaling_rule: str  # "none", "linear_wrt_256", "sqrt_wrt_64"


@dataclasses.dataclass
class WeightDecaySchedule:
    start: float
    peak: float
    end: float
    warmup_iterations: int


@dataclasses.dataclass
class MomentumSchedule:
    start: float
    peak: float
    end: float
    warmup_iterations: int


@dataclasses.dataclass
class TeacherTemperatureSchedule:
    start: float
    peak: float
    end: float
    warmup_iterations: int


@dataclasses.dataclass
class Configuration:
    output_dir: str
    seed: int = 42
    resume: bool = True
    resume_from: str | None = None  # If not None, path to checkpoint to resume from

    # Model
    model_size: str = (
        "base"  # Size-related parameters can be overridden individually below
    )
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    mlp_ratio: int | None = None
    patch_size: int = 16
    temp_size: int = 1
    reg_tokens: int = 4
    ls_init: float | None = 1e-5
    mlp_bias: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False
    proj_bias: bool = False
    sample_drop_rate: float = (
        0.0  # If > 0, each block will process (1-sample_drop_rate) of its inputs
    )

    # Positional embedding
    pos_embed_sincos: bool = True
    pos_embed_learnable: bool = False
    pos_embed_rope: bool = True
    pos_embed_side: int = 16  # For learnable
    pos_embed_per_block: bool = True  # For rope
    pos_embed_rotations: bool = True  # For rope
    pos_embed_base: float = 10  # For rope/sincos
    pos_embed_exp_range: tuple[float, float] = (-2, 2)  # For rope/sincos
    pos_embed_dtype: str = "float32"
    pos_embed_coords: str = "max"  # Normalize "each", "min", or "max"
    pos_embed_hw_box: tuple[float, float] = (
        0,
        1,
    )  # Spatial coordinates are defined on this box
    pos_embed_t_unit: float = 1.0  # Timestamps in seconds are multiplied by this value

    # DINO
    dino_loss_weight: float = 1.0
    dino_num_layers: int = 3
    dino_hidden_dim: int = 2048
    dino_bottleneck_dim: int = 256
    dino_num_prototypes: int = 65536
    dino_temp: float = 0.10

    # IBOT
    ibot_loss_weight: float = 0.5
    ibot_num_layers: int = 3
    ibot_hidden_dim: int = 2048
    ibot_bottleneck_dim: int = 256
    ibot_num_prototypes: int = 65536
    ibot_temp: float = 0.10

    # Masking
    mask_sample_ratio: float = 0.5  # How many samples per batch are masked
    mask_patch_ratio: tuple[float, float] = (
        0.1,
        0.5,
    )  # Mask between [min, max] patches in each image

    # Koleo
    koleo_loss_weight: float = 0.1

    # FSDP and efficiency
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    num_replicas: int | None = None  # Defaults to world size
    num_shards: int | None = None  # Defaults to 1
    allow_tf32: bool = True
    compile: bool = True
    checkpoint: bool = True
    profile_memory: bool = False
    torch_profiler: bool = False

    # Data
    dataset: str = "Kinetics_400:split=TRAIN"
    dataset_cache: bool = True
    num_workers: int = 8
    batch_size: int = 256
    prefetch_factor: int = 2
    in_order: bool = True

    # Spatial/temporal augmentation
    global_crop_size: int = 224
    global_crop_scale: tuple[float, float] = (0.32, 1.0)
    global_crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    global_crop_num: int = 2
    num_frames: int = 8

    local_crop_size: int = 96
    local_crop_scale: tuple[float, float] = (0.05, 0.32)
    local_crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    local_crop_num: int = 8
    local_frame_ratio: float = (
        1.0  # Local clips have this ratio of frames compared to global clips
    )

    dt_range: tuple[float, float] | None = (
        1 / 6,
        1 / 4,
    )  # Min/max timedelta between frames (seconds)
    fps_range: tuple[
        float, float
    ] | None = None  # Min/max FPS, mutually exclusive with dt_range
    variable_fps: bool = False  # If true, sample a timedelta/fps for each frame, otherwise one for each clip
    timestamp_stretch: tuple[float, float] = (9 / 10, 11 / 10)
    horizontal_flip: bool = True
    temp_crop_scale: float = (
        0.5  # Temporal cropping relative to duration of the clip (0.0 = no cropping)
    )
    # Chores
    stats_frequency: int = 50
    metrics_frequency: int = 10
    gc_frequency: int = 500
    ckpt_frequency: int = 500
    log_frequency: int = 10
    ckpt_keep_last: int | None = 3
    ckpt_keep_every: int | None = None  # If not None, checkpoints at (ckpt_keep_every * ckpt_frequency) will be kept

    # Permutation training
    permute_frames: str | None = "null"  # "null", "reverse", "random", "mixed"
    mixed_random_ratio: float = 0.5  # For "mixed", probability of applying "random"

    # Optimization
    total_iterations: int = 20_000
    clip_grad: float = 3.0
    betas: tuple[float, float] = (0.9, 0.999)
    freeze_last_layer_iterations: int = 0
    patch_embed_lr_mult: float = 1.0
    layerwise_lr_decay: float = 0.9  # Set to 1.0 to disable
    lr: LearningRateSchedule = dataclasses.field(
        default_factory=lambda: LearningRateSchedule(
            start=1e-6,
            peak=1e-3,
            end=1e-6,
            warmup_iterations=1_000,
            scaling_rule="sqrt_wrt_64",
        )
    )
    wd: WeightDecaySchedule = dataclasses.field(
        default_factory=lambda: WeightDecaySchedule(
            start=0.04,
            peak=0.04,
            end=0.4,
            warmup_iterations=0,
        )
    )
    momentum: MomentumSchedule = dataclasses.field(
        default_factory=lambda: MomentumSchedule(
            start=0.992,
            peak=0.992,
            end=1.0,
            warmup_iterations=0,
        )
    )
    temperature: TeacherTemperatureSchedule = dataclasses.field(
        default_factory=lambda: TeacherTemperatureSchedule(
            start=0.04,
            peak=0.07,
            end=0.07,
            warmup_iterations=1_000,
        )
    )

    # Fairvit evaluations
    # - If eval_frequency is None, no eval checkpoint is saved and no eval is submitted.
    # - If eval_frequency is an int, an eval checkpoint is saved every eval_frequency iterations.
    #   Also, if eval_config is not None, a fairvit benchmark is submitted for that checkpoint.
    fairvit_path: str = "../.."  # Relative to this file or absolute
    eval_config: str | None = "eval/quick.yaml"  # Relative to this file or absolute
    eval_frequency: int | None = 1_000
    eval_config_low_freq: str | None = None  # Relative to this file or absolute
    eval_frequency_low_freq: int | None = (
        None  # If not None, do an additional eval at this lower frequency
    )
    eval_qos: str | None = None
    eval_img_fps: float = 5.0
    eval_img_nf: int = 8
    eval_img_mode: str = "zoom_in"  # "repeat" or "zero_pad" or "zoom_in"
    eval_zoom_ratio: float = 0.5  # For "zoom_in" mode
    eval_use_real_timestamps: bool = False


class SinCosAdditivePositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_axes: int,
        base: float,
        exp_range: tuple[float, float],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.base = base
        self.exp_range = exp_range
        num_periods = embed_dim // (2 * num_axes)
        self.padding = embed_dim % (2 * num_axes)
        self.periods: Tensor
        self.register_buffer(
            "periods",
            torch.empty(num_periods, device=device, dtype=dtype),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        dd = {"device": self.periods.device, "dtype": self.periods.dtype}
        exponents = torch.linspace(
            *self.exp_range, self.periods.shape[0], **dd
        )  # [num_periods]
        periods = self.base**exponents  # [num_periods]
        self.periods.data[:] = periods

    @property
    def dtype(self) -> torch.dtype:
        return self.periods.dtype

    def forward(
        self,
        coords: Tensor,  # [B, L, num_axes]
    ) -> Tensor:
        # It's recommended to create the coords with the same dtype as periods, but we cast here just in case
        B, L, num_axes = coords.shape
        coords = coords.to(dtype=self.dtype)

        angles = (
            2 * math.pi * coords[..., None] / self.periods
        )  # [B, L, num_axes, num_periods]
        angles = angles.flatten(2, 3)  # [B, L, num_axes * num_periods]
        sin = torch.sin(angles)  # [B, L, num_axes * num_periods]
        cos = torch.cos(angles)  # [B, L, num_axes * num_periods]
        sincos = torch.cat([sin, cos], dim=-1)  # [B, L, 2 * num_axes * num_periods]
        if self.padding > 0:
            sincos = F.pad(
                sincos, (0, self.padding), mode="constant", value=0.0
            )  # [B, L, D]
        sincos = sincos / math.sqrt(self.embed_dim)
        return sincos


class RopeAxial(nn.Module):
    def __init__(
        self,
        D_head: int,
        num_heads: int,
        num_axes: int,
        base: Sequence[
            float
        ],  # Length 1 (same for all heads) or num_heads (different base for each head)
        exp_range: tuple[float, float],
        rotations: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.D_head = D_head
        self.num_heads = num_heads
        self.num_axes = num_axes

        if len(base) not in {1, num_heads}:
            raise ValueError(f"Invalid {len(base)=}, expected 1 or {num_heads=}")
        self.base = base
        self.exp_range = exp_range
        num_periods = D_head // (2 * num_axes)
        self.padding = D_head % (2 * num_axes)
        self.weight: Tensor
        weight = torch.empty(
            (len(base), num_periods), dtype=dtype, device=device
        )  # Use len(base) to broadcast
        self.register_buffer("weight", weight, persistent=True)

        self.rotations: Tensor | None
        if rotations and num_axes > 1:
            rot = torch.empty(
                (num_heads, num_axes, num_axes), dtype=dtype, device=device
            )
            self.register_buffer("rotations", rot, persistent=True)
        else:
            self.rotations = None

        self.reset_parameters()

    def reset_parameters(self):
        dd = {"device": self.weight.device, "dtype": self.weight.dtype}

        base = torch.as_tensor(self.base, **dd)  # [num_heads], num_heads might be 1
        exponents = torch.linspace(
            *self.exp_range, self.D_head // (2 * self.num_axes), **dd
        )  # [D_head//(2*num_axes)]
        weight = (
            2
            * torch.pi
            * torch.pow(
                base[:, None],  # [num_heads, 1]
                -exponents[None, :],  # [1, D_head//(2*num_axes)]
            )
        )  # [num_heads, D_head // (2 * num_axes)], num_heads might be 1
        self.weight.data[:] = weight

        if self.rotations is not None:
            if self.num_axes == 1:
                # 1D, no rotation, this case should never happen, keep it for completeness
                nn.init.ones_(self.rotations)  # [num_heads, 1, 1]
            elif self.num_axes == 2:
                # 2D, rotations around the origin by random angles
                # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
                angles = 2 * torch.pi * torch.rand(self.num_heads, **dd)  # [num_heads]
                self.rotations.data[:] = torch.stack(
                    [
                        torch.stack([torch.cos(angles), -torch.sin(angles)], dim=-1),
                        torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1),
                    ],
                    dim=1,
                )  # [num_heads, 2, 2]
            elif self.num_axes == 3:
                # 3D, rotations around the origin by random angles
                # https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
                alpha, beta, gamma = (
                    2 * torch.pi * torch.rand(3, self.num_heads, **dd)
                )  # 3 * [num_heads]
                zero = torch.zeros_like(alpha)
                one = torch.ones_like(alpha)
                yaw = torch.stack(
                    [
                        torch.stack([alpha.cos(), -alpha.sin(), zero], dim=-1),
                        torch.stack([alpha.sin(), alpha.cos(), zero], dim=-1),
                        torch.stack([zero, zero, one], dim=-1),
                    ],
                    dim=1,
                )
                pitch = torch.stack(
                    [
                        torch.stack([beta.cos(), zero, beta.sin()], dim=-1),
                        torch.stack([zero, one, zero], dim=-1),
                        torch.stack([-beta.sin(), zero, beta.cos()], dim=-1),
                    ],
                    dim=1,
                )
                roll = torch.stack(
                    [
                        torch.stack([one, zero, zero], dim=-1),
                        torch.stack([zero, gamma.cos(), -gamma.sin()], dim=-1),
                        torch.stack([zero, gamma.sin(), gamma.cos()], dim=-1),
                    ],
                    dim=1,
                )
                self.rotations.data[:] = torch.einsum(
                    "hij,hjk,hkl->hil", yaw, pitch, roll
                )  # [num_heads, 3, 3]
            else:
                raise ValueError(
                    f"Random rotations supported up to 3 axes, got {self.num_axes=}"
                )

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(
        self,
        coords: Tensor,  # [B, L, num_axes]
    ) -> Tensor:
        # It's recommended to create the coords with the same dtype as periods, but we cast here just in case
        B, L, num_axes = coords.shape
        coords = coords.to(dtype=self.dtype)

        if self.rotations is None:
            coords = coords[:, :, None, :]  # [B, L, 1, num_axes]
        else:
            coords = torch.einsum(
                "hij, blj -> blhi", self.rotations, coords
            )  # [B, L, num_heads, num_axes]

        P = self.weight.shape[-1]  # num_periods = D_head // (2 * num_axes)
        angles = torch.multiply(
            coords[:, :, :, :, None],  #            [B, L, num_heads, num_axes, 1]
            self.weight[None, None, :, None, :],  # [1, 1, num_heads,        1, P]
        )  # [B, L, num_heads, num_axes, P]
        angles = angles.repeat_interleave(
            2, dim=-1, output_size=P * 2
        )  # [B, L, num_heads, num_axes, P * 2]
        angles = angles.flatten(
            3, 4
        )  #                                   [B, L, num_heads, num_axes * P * 2]

        # Note: at this point, the angles for axes a, b, c are in this order:
        # [a0 a0 a1 a1 a2 a2 ... b0 b0 b1 b1 b2 b2 ... c0 c0 c1 c1 c2 c2 ...]

        if self.padding > 0:
            angles = F.pad(
                angles, (0, self.padding), mode="constant", value=0.0
            )  # [B, L, num_heads, D_head]

        cos = torch.cos(angles)  # [B, L, num_heads, D_head]
        sin = torch.sin(angles)  # [B, L, num_heads, D_head]
        rope = torch.stack([cos, sin], dim=0)  # [2, B, L, num_heads, D_head]

        return rope


class DinoVideoPretraining(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        size_dict = VIT_SIZES[cfg.model_size]
        backbone = VidTransformer(
            embed_dim=size_dict["embed_dim"]
            if cfg.embed_dim is None
            else cfg.embed_dim,
            depth=size_dict["depth"] if cfg.depth is None else cfg.depth,
            num_heads=size_dict["num_heads"]
            if cfg.num_heads is None
            else cfg.num_heads,
            mlp_ratio=size_dict["mlp_ratio"]
            if cfg.mlp_ratio is None
            else cfg.mlp_ratio,
            mlp_bias=cfg.mlp_bias,
            patch_size=cfg.patch_size,
            temp_size=cfg.temp_size,
            reg_tokens=cfg.reg_tokens,
            ls_init=cfg.ls_init,
            sample_drop_rate=cfg.sample_drop_rate,
            qkv_bias=cfg.qkv_bias,
            qk_norm=cfg.qk_norm,
            proj_bias=cfg.proj_bias,
            pos_embed_sincos=cfg.pos_embed_sincos,
            pos_embed_learnable=cfg.pos_embed_learnable,
            pos_embed_rope=cfg.pos_embed_rope,
            pos_embed_side=cfg.pos_embed_side,
            pos_embed_per_block=cfg.pos_embed_per_block,
            pos_embed_rotations=cfg.pos_embed_rotations,
            pos_embed_base=cfg.pos_embed_base,
            pos_embed_exp_range=cfg.pos_embed_exp_range,
            pos_embed_dtype=cfg.pos_embed_dtype,
            pos_embed_coords=cfg.pos_embed_coords,
            pos_embed_hw_box=cfg.pos_embed_hw_box,
        )
        dino_head = Head(
            in_dim=size_dict["embed_dim"] if cfg.embed_dim is None else cfg.embed_dim,
            hidden_dim=cfg.dino_hidden_dim,
            bottleneck_dim=cfg.dino_bottleneck_dim,
            num_prototypes=cfg.dino_num_prototypes,
            num_layers=cfg.dino_num_layers,
            mlp_bias=cfg.mlp_bias,
        )
        ibot_head = Head(
            in_dim=size_dict["embed_dim"] if cfg.embed_dim is None else cfg.embed_dim,
            hidden_dim=cfg.dino_hidden_dim,
            bottleneck_dim=cfg.dino_bottleneck_dim,
            num_prototypes=cfg.dino_num_prototypes,
            num_layers=cfg.dino_num_layers,
            mlp_bias=cfg.mlp_bias,
        )
        self.student = BackboneWithHeads(backbone, dino_head, ibot_head)
        self.teacher = copy.deepcopy(self.student)

        self.dino_temp = cfg.dino_temp
        self.ibot_temp = cfg.ibot_temp

        # Prepare for training
        self.train()
        self.requires_grad_()

    def train(self, mode: bool = True):
        self.teacher.train(False)
        self.student.train(mode)

    def requires_grad_(self, requires_grad: bool = True):
        self.teacher.requires_grad_(False)
        self.student.requires_grad_(requires_grad)

    def init_weights(self):
        """Initialize weights from scratch for the whole model, not just this module."""
        # self.student.apply(nap.torch.init_weights)
        # nap.torch.ema_init(self.student, self.teacher)
        pass

    def forward(
        self,
        global_crops: Tensor,  # [B, global_crop_num, T, 3, H, W]
        local_crops: Tensor,  # [B, local_crop_num, T, 3, H, W]
        timestamps_global: Tensor,  # [B, global_crop_num, T], for temporal pos embedding
        timestamps_local: Tensor,  # [B, local_crop_num, T], for temporal pos embedding
        mask_bool: Tensor,  # [B, global_crop_num, T, h, w]
        mask_nonzero: Tensor,  # [num_masks]
        mask_weight: Tensor,  # [num_masks]
        temperature: float,
        with_metrics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        teacher_cls_probs, teacher_patch_probs, teacher_metrics = self.forward_teacher(
            global_crops,
            timestamps_global,
            mask_nonzero,
            temperature,
            with_metrics,
        )
        (
            student_cls_feats,
            student_cls_logits,
            student_patch_logits,
            student_metrics,
        ) = self.forward_student(
            global_crops,
            local_crops,
            timestamps_global,
            timestamps_local,
            mask_bool,
            mask_nonzero,
            with_metrics,
        )
        loss, loss_metrics = self.forward_losses(
            teacher_cls_probs,
            teacher_patch_probs,
            student_cls_feats,
            student_cls_logits,
            student_patch_logits,
            mask_weight,
        )
        return loss  # , teacher_metrics | student_metrics | loss_metrics

    @torch.no_grad()
    def forward_teacher(
        self,
        global_crops: Tensor,  # [B, global_crop_num, T, 3, H, W]
        timestamps_global: Tensor,  # [B, global_crop_num, T]
        mask_nonzero: Tensor,  # [num_masks]
        temperature: float,
        with_metrics: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        B, global_crop_num, _, _, H, W = global_crops.shape
        metrics = {}

        # Backbone
        cls_feats, reg_feats, patch_feats = self.teacher.backbone(
            global_crops.flatten(0, 1), timestamps_global.flatten(0, 1)
        )
        cls_feats = cls_feats.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, D]
        reg_feats = reg_feats.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, R, D]
        patch_feats = patch_feats.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, T, h, w, D]

        # DINO head on CLS tokens
        cls_logits = self.teacher.dino_head(
            cls_feats.flatten(0, 1)
        ).float()  # [B * global_crop_num, K]
        cls_probs = sinkhorn_knopp(cls_logits, temperature)  # [B * global_crop_num, K]
        if with_metrics:
            metrics["teacher_cls_avg_entropy"] = (
                torch.distributions.Categorical(probs=cls_probs).entropy().mean()
            )
            metrics["teacher_cls_entropy_avg"] = torch.distributions.Categorical(
                probs=cls_probs.mean(0)
            ).entropy()
        cls_probs = cls_probs.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, K]

        # IBOT head on patches that are masked for the student
        patch_logits = self.teacher.ibot_head(
            patch_feats.flatten(0, 4)
            .unflatten(0, (mask_nonzero.shape[0], -1))
            .gather(1, mask_nonzero[..., None].expand(-1, -1, patch_feats.shape[-1]))
        ).float()  # [num_gpu, num_masks, K]
        patch_probs = sinkhorn_knopp(patch_logits, temperature)  # [num_masks, K]
        if with_metrics:
            metrics["teacher_patch_avg_entropy"] = (
                torch.distributions.Categorical(probs=patch_probs).entropy().mean()
            )
            metrics["teacher_patch_entropy_avg"] = torch.distributions.Categorical(
                probs=patch_probs.mean(0)
            ).entropy()

        return cls_probs, patch_probs, metrics

    def forward_student(
        self,
        global_crops: Tensor,  # [B, global_crop_num, T, 3, H, W]
        local_crops: Tensor,  # [B, local_crop_num, T, 3, H', W']
        timestamps_global: Tensor,  # [B, global_crop_num, T]
        timestamps_local: Tensor,  # [B, local_crop_num, T]
        mask_bool: Tensor,  # [B, global_crop_num, T, h, w]
        mask_nonzero: Tensor,  # [num_masks]
        with_metrics: bool,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        B, global_crop_num, _, _, H, W = global_crops.shape
        B, local_crop_num, _, _, _, _ = local_crops.shape
        crop_num = global_crop_num + local_crop_num
        metrics = {}

        # Backbone global crops
        global_crops = global_crops.flatten(0, 1)  # [B * global_crop_num, T, 3, H, W]
        mask_bool = mask_bool.flatten(0, 1)  # [B * global_crop_num, T, h, w]
        timestamps_global = timestamps_global.flatten(0, 1)  # [B * global_crop_num, T]

        cls_feats_global, _, patch_feats_global = self.student.backbone(
            global_crops, timestamps_global, mask_bool
        )
        cls_feats_global = cls_feats_global.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, D]
        patch_feats_global = patch_feats_global.unflatten(
            0, (B, global_crop_num)
        )  # [B, global_crop_num, T, h, w, D]

        # Backbone local crops
        local_crops = local_crops.flatten(0, 1)  # [B * local_crop_num, T, 3, H', W']
        timestamps_local = timestamps_local.flatten(0, 1)  # [B * local_crop_num, T]

        cls_feats_local, _, _ = self.student.backbone(local_crops, timestamps_local)
        cls_feats_local = cls_feats_local.unflatten(
            0, (B, local_crop_num)
        )  # [B, local_crop_num, D]

        # DINO head on CLS tokens (global and local)
        x = torch.cat([cls_feats_global, cls_feats_local], dim=1)  # [B, crop_num, D]
        x = x.flatten(0, 1)  # [B * crop_num, D]
        cls_logits = self.student.dino_head(x).float()  # [B * crop_num, K]
        if with_metrics:
            cls_probs = (cls_logits.detach() / self.dino_temp).softmax(-1)
            metrics["student_cls_avg_entropy"] = (
                torch.distributions.Categorical(probs=cls_probs).entropy().mean()
            )
            metrics["student_cls_entropy_avg"] = torch.distributions.Categorical(
                probs=cls_probs.mean(0)
            ).entropy()
        cls_logits = cls_logits.unflatten(0, (B, crop_num))  # [B, crop_num, K]

        # IBOT head on masked patches
        x = (
            patch_feats_global.flatten(0, 4)
            .unflatten(0, (mask_nonzero.shape[0], -1))
            .gather(
                1, mask_nonzero[..., None].expand(-1, -1, patch_feats_global.shape[-1])
            )
        )  # [ngpu, num_masks, D]
        patch_logits_global = self.student.ibot_head(x).float()  # [num_masks, K]
        if with_metrics:
            patch_probs = (patch_logits_global.detach() / self.ibot_temp).softmax(-1)
            metrics["student_patch_avg_entropy"] = (
                torch.distributions.Categorical(probs=patch_probs).entropy().mean()
            )
            metrics["student_patch_entropy_avg"] = torch.distributions.Categorical(
                probs=patch_probs.mean(0)
            ).entropy()

        return cls_feats_global, cls_logits, patch_logits_global, metrics

    def forward_losses(
        self,
        teacher_cls_probs,  # [B, global_crop_num, K]
        teacher_patch_probs,  # [num_masks, K]
        student_cls_feats,  # [B, global_crop_num, D]
        student_cls_logits,  # [B, global_crop_num + local_crop_num, K]
        student_patch_logits,  # [num_masks, K]
        mask_weight,  # [num_masks]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss = 0.0
        loss_dict = {}

        # dino_loss = nap.torch.pairwise_cross_entropy(student_cls_logits / self.dino_temp, teacher_cls_probs)
        (
            dino_loss,
            dino_loss_global,
            dino_loss_local,
        ) = pairwise_cross_entropy_global_and_local(
            student_cls_logits / self.dino_temp, teacher_cls_probs
        )
        loss_dict["dino_loss"] = dino_loss.detach()
        loss_dict["dino_loss_global"] = dino_loss_global.detach()  # log for debug
        loss_dict["dino_loss_local"] = dino_loss_local.detach()
        loss += self.cfg.dino_loss_weight * dino_loss

        ibot_loss = cross_entropy(
            student_patch_logits / self.ibot_temp, teacher_patch_probs
        )
        ibot_loss = torch.sum(ibot_loss * mask_weight)  # [num_masks]
        loss_dict["ibot_loss"] = ibot_loss.detach()
        loss += self.cfg.ibot_loss_weight * ibot_loss

        koleo_loss = koleo_batched(student_cls_feats.transpose(0, 1))
        loss_dict["koleo_loss"] = koleo_loss.detach()
        loss += self.cfg.koleo_loss_weight * koleo_loss

        return loss, loss_dict


class VidTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        temp_size: int,
        reg_tokens: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        sample_drop_rate: float,
        qkv_bias: bool,
        qk_norm: bool,
        proj_bias: bool,
        mlp_ratio: int,
        mlp_bias: bool,
        ls_init: float | None,
        pos_embed_sincos: bool,
        pos_embed_learnable: bool,
        pos_embed_rope: bool,
        pos_embed_side: int,
        pos_embed_per_block: bool,
        pos_embed_rotations: bool,
        pos_embed_base: float,
        pos_embed_exp_range: tuple[float, float],
        pos_embed_dtype: Literal["float64", "float32", "float16", "bfloat16"],
        pos_embed_coords: Literal["each", "min", "max"],
        pos_embed_hw_box: tuple[float, float],
        pos_embed_t_unit: float = 1.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"{embed_dim=} must be divisible by {num_heads=}")
        self.embed_dim = embed_dim
        self.D_head = embed_dim // num_heads

        # Patch embedding
        self.patch_size = patch_size
        self.temp_size = temp_size
        self.patch_embed = nn.Linear(
            temp_size * 3 * patch_size**2, embed_dim, bias=True
        )

        # Tokens
        self.cls_token = nn.Parameter(torch.empty(embed_dim))
        self.reg_tokens = nn.Parameter(torch.empty(reg_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(embed_dim))

        # Patch & Time coordinates
        self.pos_embed_coords = pos_embed_coords
        self.pos_embed_hw_box = pos_embed_hw_box
        self.pos_embed_dtype = DTYPES[pos_embed_dtype]
        self.pos_embed_t_unit = pos_embed_t_unit

        # Additive positional encoding (sincos)
        self.pos_embed_sincos: SinCosAdditivePositionalEncoding | None
        if pos_embed_sincos:
            self.pos_embed_sincos = SinCosAdditivePositionalEncoding(
                embed_dim,
                num_axes=3,
                base=pos_embed_base,
                exp_range=pos_embed_exp_range,
                dtype=self.pos_embed_dtype,
            )
        else:
            self.pos_embed_sincos = None

        # Additive positional encoding (learnable)
        # self.pos_embed_learnable: PE.LearnablePositionalEncoding | None
        if pos_embed_learnable:
            self.pos_embed_learnable = PE.LearnablePositionalEncoding(
                embed_dim, pos_embed_side
            )
        else:
            self.pos_embed_learnable = None

        # Rotary positional encoding
        self.pos_embed_rope: RopeAxial3DRotate2D | nn.ModuleList | None
        if pos_embed_rope:
            if pos_embed_per_block:
                self.pos_embed_rope = nn.ModuleList(
                    [
                        RopeAxial3DRotate2D(
                            self.D_head,
                            num_heads=num_heads,
                            base=(pos_embed_base,),
                            exp_range=pos_embed_exp_range,
                            rotations=pos_embed_rotations,
                            dtype=self.pos_embed_dtype,
                        )
                        for _ in range(depth)
                    ]
                )
            else:
                self.pos_embed_rope = RopeAxial3DRotate2D(
                    self.D_head,
                    num_heads=num_heads,
                    base=(pos_embed_base,),
                    exp_range=pos_embed_exp_range,
                    rotations=pos_embed_rotations,
                    dtype=self.pos_embed_dtype,
                )
        else:
            self.pos_embed_rope = None

        # Transformer blocks
        blk = partial(
            SelfBlock,
            embed_dim,
            attn_layer=partial(
                SelfAttention,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
            ),
            ffn_layer=partial(
                SwiGLUFFN,
                hidden_features=round(embed_dim * mlp_ratio),
                bias=mlp_bias,
            ),
            ls_init=ls_init,
            sample_drop_rate=sample_drop_rate,
        )
        self.n_blocks = depth
        self.blocks = nn.ModuleList([blk() for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.reg_tokens, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        vid: Tensor,  # [B, T, 3, H, W]
        timestamps: Tensor,  # [B, T]
        mask_bool: Tensor
        | None = None,  # [B, T, H, W] True will be masked, False will be visible
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, rgb, H, W = vid.shape
        dtype = vid.dtype

        # Patch embedding
        patches = patchify_3d(
            vid, self.patch_size, self.temp_size
        )  # [B, t, h, w, V3PP]
        timestamps = timestamps[:, :: self.temp_size]  # [B, t]

        _, t, h, w, _ = patches.shape
        patches = patches.flatten(1, 3)  # [B, thw, V3PP]
        patches = self.patch_embed(patches)  # [B, thw, D]
        coords_thw = self._get_coords(timestamps, h, w).flatten(1, 3)  # [B, thw, 3]

        # Masking
        if mask_bool is not None:
            patches = torch.where(
                mask_bool.flatten(1, 3).unsqueeze(-1), self.mask_token, patches
            )

        # Additive positional embeddings
        if self.pos_embed_sincos is not None:
            pos_embed = self.pos_embed_sincos(coords_thw)  # [B, thw, D]
            patches = patches + pos_embed.to(dtype)  # [B, thw, D]

        # Rotary positional embeddings
        if isinstance(self.pos_embed_rope, RopeAxial):
            rope = self.pos_embed_rope(coords_thw)  # [2, B, thw, num_heads, D_head]
        else:
            rope = None

        # Concat extra tokens
        cls_token = self.cls_token.expand(B, 1, -1)
        reg_tokens = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat([cls_token, reg_tokens, patches], dim=1)  # [B, extra + thw, D]

        # Transformer blocks
        for i, blk in enumerate(self.blocks):
            if isinstance(self.pos_embed_rope, nn.ModuleList):
                rope = self.pos_embed_rope[i](
                    coords_thw
                )  # [2, B, thw, num_heads, D_head]
            x = blk(x, rope=rope)
        x = self.norm(x)

        # Split extra tokens
        cls_token, reg_tokens, patches = torch.split_with_sizes(
            x, [1, self.reg_tokens.shape[0], t * h * w], dim=1
        )
        cls_token = cls_token.squeeze(1)  # [B, D]
        patches = patches.unflatten(1, (t, h, w))  # [B, t, h, w, D]

        return cls_token, reg_tokens, patches

    def _get_coords(
        self,
        timestamps: Tensor,  # [B, T]
        h: int,
        w: int,
    ) -> Tensor:
        B, T = timestamps.shape
        coords_t = self.pos_embed_t_unit * timestamps.to(
            dtype=self.pos_embed_dtype
        )  # [B, T]
        coords_hw = coords_relative_2d(
            h,
            w,
            normalize=self.pos_embed_coords,
            box=self.pos_embed_hw_box,
            device=timestamps.device,
            dtype=self.pos_embed_dtype,
        )  # [h, w, 2]
        coords_thw = torch.cat(
            [
                coords_t[:, :, None, None, None].expand(
                    -1, -1, h, w, 1
                ),  # [B, T, h, w, 1]
                coords_hw[None, None, :, :, :].expand(
                    B, T, -1, -1, -1
                ),  #  [B, T, h, w, 2]
            ],
            dim=-1,
        )  # [B, T, h, w, 3]
        return coords_thw


class Head(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_prototypes: int,
        num_layers: int,
        mlp_bias: bool,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [bottleneck_dim]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out, bias=mlp_bias))
            layers.append(nn.GELU())
        layers.pop(-1)
        layers.append(L2Norm())
        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, num_prototypes, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return self.last_layer(x)


def coords_relative_2d(
    h: int,
    w: int,
    normalize: Literal["each", "min", "max"],
    box: tuple[float, float],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    coords_h = torch.arange(0.5, h, device=device, dtype=dtype)
    coords_w = torch.arange(0.5, w, device=device, dtype=dtype)

    if normalize == "min":
        coords_h /= min(h, w)
        coords_w /= min(h, w)
    elif normalize == "max":
        coords_h /= max(h, w)
        coords_w /= max(h, w)
    elif normalize == "each":
        coords_h /= h
        coords_w /= w
    else:
        raise ValueError(f"Unknown {normalize=}")

    grid = torch.meshgrid(coords_h, coords_w, indexing="ij")
    grid = torch.stack(grid, dim=-1)  # [h, w, 2]
    grid = (
        grid * (box[1] - box[0]) + box[0]
    )  # When h==w, coordinates are defined on the range box^2
    return grid


class RopeAxial3DRotate2D(RopeAxial):
    def __init__(
        self,
        D_head: int,
        num_heads: int,
        base: Sequence[
            float
        ],  # Length 1 (same for all heads) or num_heads (different base for each head)
        exp_range: tuple[float, float],
        rotations: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            D_head,
            num_heads=num_heads,
            num_axes=3,
            base=base,
            exp_range=exp_range,
            rotations=rotations,
            dtype=dtype,
            device=device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        dd = {"device": self.weight.device, "dtype": self.weight.dtype}

        if self.rotations is not None:
            # Assume coordinates (t, i, j) where t is time, i is height, j is width.
            # Rotate (i, j), leave t unchanged.
            angles = 2 * torch.pi * torch.rand(self.num_heads, **dd)  # [num_heads]
            zero = torch.zeros_like(angles)
            one = torch.ones_like(angles)
            self.rotations.data[:] = torch.stack(
                [
                    torch.stack([one, zero, zero], dim=-1),
                    torch.stack([zero, angles.cos(), -angles.sin()], dim=-1),
                    torch.stack([zero, angles.sin(), angles.cos()], dim=-1),
                ],
                dim=1,
            )


class BackboneWithHeads(nn.Module):
    # This could have been a nn.ModuleDict, but with this implementation we get type hints
    def __init__(self, backbone: VidTransformer, dino_head: Head, ibot_head: Head):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head


def patchify_3d(x: Tensor, patch_size: int, temp_size: int) -> Tensor:
    # x: [B, T, C, H, W]
    B, T, C, H, W = x.shape
    P = patch_size
    V = temp_size
    x = x.reshape(B, T // V, V, C, H // P, P, W // P, P)  # [B, t, V, C, h, P, w, P]
    x = x.movedim((2, 3, 5, 7), (4, 5, 6, 7))  # [B, t, h, w, V, C, P, P]
    x = x.flatten(4, -1)  # [B, t, h, w, VCPP]
    return x


def unpatchify_3d(x: Tensor, patch_size: int, temp_size: int) -> Tensor:
    # x: [B, t, h, w, VCPP]
    B, t, h, w, VCPP = x.shape
    P = patch_size
    V = temp_size
    C = VCPP // (V * P * P)
    x = x.unflatten(4, (V, C, P, P))  # [B, t, h, w, V, C, P, P]
    x = x.movedim((4, 5, 6, 7), (2, 3, 5, 7))  # [B, t, V, C, h, P, w, P]
    x = x.reshape(B, t * V, C, h * P, w * P)  # [B, T, C, H, W]
    return x


def pairwise_cross_entropy_global_and_local(
    logits: Tensor,  # [B, student_crop_num, K]
    probs: Tensor,  # [B, teacher_crop_num, K]
) -> Tensor:
    B, student_num, K = logits.shape
    _, teacher_num, _ = probs.shape
    logits = logits.float()
    probs = probs.float()
    log_prob = F.log_softmax(logits, dim=-1)
    # Same as: ce = -torch.einsum("bsk, btk -> ", log_prob, probs) / (B * student_num * teacher_num)
    ce = -torch.mul(
        log_prob[:, :, None, :],  # [B, student_crop_num, 1, K]
        probs[:, None, :, :],  # [B, 1, teacher_crop_num, K]
    ).sum(
        -1
    )  # [B, student_crop_num, teacher_crop_num]

    ce_global = ce[:, :teacher_num, :].mean()  # [B, teacher_crop_num, teacher_crop_num]
    ce_local = ce[
        :, teacher_num:, :
    ].mean()  # [B, student_crop_num - teacher_crop_num, teacher_crop_num]
    ce = ce.mean()
    return ce, ce_global, ce_local


class ExtraKvInit(enum.Enum):
    LEARN = enum.auto()
    ZERO = enum.auto()
    RAND = enum.auto()


class AttnImpl(enum.Enum):
    SLOW = enum.auto()
    TORCH = enum.auto()
    XFORMERS = enum.auto()
    FLEX = enum.auto()


AttnMaskType = Tensor | None


class ExtraKvMixin:
    num_extra_kv: int
    extra_k_init: ExtraKvInit
    extra_v_init: ExtraKvInit

    def _init_extra_kv(
        self,
        num_extra_kv: int,
        extra_k_init: ExtraKvInit,
        extra_v_init: ExtraKvInit,
        num_heads: int,
        D_head: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ):
        self.num_extra_kv = num_extra_kv
        self.extra_k_init = extra_k_init
        self.extra_v_init = extra_v_init
        dd = {"device": device, "dtype": dtype}
        if num_extra_kv > 0:
            extra_kv_shape = (num_extra_kv, num_heads, D_head)
            if extra_k_init is ExtraKvInit.LEARN:
                self.extra_k = nn.Parameter(torch.empty(extra_kv_shape, **dd))
            elif extra_k_init in {ExtraKvInit.ZERO, ExtraKvInit.RAND}:
                self.register_buffer(
                    "extra_k", torch.empty(extra_kv_shape, **dd), persistent=True
                )
            else:
                raise ValueError(f"Unknown {extra_k_init=}")
            if extra_v_init is ExtraKvInit.LEARN:
                self.extra_v = nn.Parameter(torch.empty(extra_kv_shape, **dd))
            elif extra_v_init in {ExtraKvInit.ZERO, ExtraKvInit.RAND}:
                self.register_buffer(
                    "extra_v", torch.empty(extra_kv_shape, **dd), persistent=True
                )
            else:
                raise ValueError(f"Unknown {extra_v_init=}")

    def _reset_extra_kv(self):
        if self.num_extra_kv > 0:
            if self.extra_k_init in {ExtraKvInit.LEARN, ExtraKvInit.RAND}:
                nn.init.trunc_normal_(self.extra_k, std=0.02)
            elif self.extra_k_init is ExtraKvInit.ZERO:
                nn.init.zeros_(self.extra_k)
            else:
                raise ValueError(f"Unknown {self.extra_k_init=}")
            if self.extra_v_init in {ExtraKvInit.LEARN, ExtraKvInit.RAND}:
                nn.init.trunc_normal_(self.extra_v, std=0.02)
            elif self.extra_v_init is ExtraKvInit.ZERO:
                nn.init.zeros_(self.extra_v)
            else:
                raise ValueError(f"Unknown {self.extra_v_init=}")

    def _apply_extra_kv(
        self,
        k: Tensor,  # [B, M, head, D_head]
        v: Tensor,  # [B, M, head, D_head]
        attn_mask: AttnMaskType,  # [N, M] or [B, head, N, M] or None
    ) -> tuple[Tensor, Tensor, AttnMaskType]:
        if self.num_extra_kv > 0:
            B = k.shape[0]
            k = torch.cat(
                [k, self.extra_k.expand(B, -1, -1, -1)], dim=2
            )  # [B, M + num_extra_kv, head, D_head]
            v = torch.cat(
                [v, self.extra_v.expand(B, -1, -1, -1)], dim=2
            )  # [B, M + num_extra_kv, head, D_head]
            if attn_mask is None:
                pass
            elif isinstance(attn_mask, X.AttentionBias):
                raise NotImplementedError(
                    "Extra KV tokens with AttentionBias not implemented"
                )
            elif isinstance(attn_mask, FA.BlockMask):
                raise NotImplementedError(
                    "Extra KV tokens with BlockMask not implemented"
                )
            elif isinstance(attn_mask, Tensor):
                attn_mask = F.pad(
                    attn_mask,
                    (0, self.num_extra_kv),
                    mode="constant",
                    value=True if attn_mask.dtype == torch.bool else 0.0,
                )  # [N, M + num_extra_kv] or [B, head, N, M + num_extra_kv]
            else:
                raise ValueError(f"Unknown {attn_mask=}")
        return k, v, attn_mask


def _attention(
    q: Tensor,  # [B, N, head, D_head]
    k: Tensor,  # [B, M, head, D_head]
    v: Tensor,  # [B, M, head, D_head]
    attn_mask: AttnMaskType,  # [N, M] or [B, head, N, M]
    drop: float,
    impl: AttnImpl,
):
    if impl is AttnImpl.SLOW:
        x, _ = slow_attention(q, k, v, attn_mask, p=drop)
    elif impl is AttnImpl.TORCH:
        x = scaled_dot_product_attention(q, k, v, attn_mask, p=drop)
    elif impl is AttnImpl.XFORMERS:
        x = memory_efficient_attention(q, k, v, attn_mask, p=drop)
    elif impl is AttnImpl.FLEX:
        x = flex_attention(q, k, v, attn_mask, p=drop)
    else:
        raise ValueError(f"Unknown {impl=}")
    return x  # [B, N, head, D_head]


def scaled_dot_product_attention(
    q: Tensor,  # [B, N, head, D_head]
    k: Tensor,  # [B, M, head, D_head]
    v: Tensor,  # [B, M, head, D_head]
    attn_mask: Tensor
    | None = None,  # [N, M] or [B, head, N, M], True is visible, False is masked
    p: float = 0.0,  # Dropout
):
    """Pytorch's scaled dot product attention, can be compiled, attn_mask must be a float/bool tensor."""
    q = q.transpose(-3, -2)  # [B, head, N, D_head]
    k = k.transpose(-3, -2)  # [B, head, M, D_head]
    v = v.transpose(-3, -2)  # [B, head, M, D_head]
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=p
    )  # [B, head, N, D_head]
    out = out.transpose(-3, -2)  # [B, N, head, D_head]
    return out


class SelfAttention(nn.Module, ExtraKvMixin):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_extra_kv: int = 0,
        extra_k_init: ExtraKvInit = ExtraKvInit.LEARN,
        extra_v_init: ExtraKvInit = ExtraKvInit.LEARN,
        attn_impl: AttnImpl = AttnImpl.TORCH,  # AttnImpl.XFORMERS,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        # Dimensions
        if dim % num_heads != 0:
            raise ValueError(f"{dim=} must be divisible by {num_heads=}")
        self.num_heads = num_heads
        self.D_head = dim // num_heads

        # QKV proj and output proj
        self.qkv = QkvLinear(dim, 3 * dim, bias=qkv_bias, **dd)
        self.attn_drop = attn_drop
        self.q_norm = (
            nn.RMSNorm(self.D_head, eps=1e-5, **dd) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.D_head, eps=1e-5, **dd) if qk_norm else nn.Identity()
        )
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

        # Extra KV tokens
        self._init_extra_kv(
            num_extra_kv, extra_k_init, extra_v_init, num_heads, self.D_head, **dd
        )

        self.attn_impl = attn_impl
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_extra_kv()

    def forward(
        self,
        x,  # [B, N, D]
        *,
        attn_mask: AttnMaskType = None,  # [N, N] or [B, head, N, N]
        rope: Tensor | None = None,  # [2, B, N', head, D_head] with N' <= N
        cache: None = None,  # Updated inplace
    ) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).unflatten(
            -1, (3, self.num_heads, self.D_head)
        )  # [B, N, 3, head, D_head]
        q, k, v = torch.unbind(qkv, -3)  # 3 * [B, N, head, D_head]

        # QK norm before RoPE
        q = self.q_norm(q)  # [B, N, head, D_head]
        k = self.k_norm(k)  # [B, N, head, D_head]

        # Apply RoPE ignoring tokens at the front for which we don't have RoPE angles.
        if rope is not None:
            q = rope_apply_with_prefix(q, rope)
            k = rope_apply_with_prefix(k, rope)

        if cache is not None:
            k, v = cache.update_(k, v)

        # Append extra KV vectors, pad the attention mask if necessary
        k, v, attn_mask = self._apply_extra_kv(k, v, attn_mask)

        # Attention
        drop = self.attn_drop if self.training else 0.0
        x = _attention(q, k, v, attn_mask, drop, self.attn_impl)  # [B, N, head, D_head]
        x = x.flatten(-2)  # [B, N, D]
        x = self.proj_drop(self.proj(x))  # [B, N, D]
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = False,
        align_to: int = 256,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = in_features
        hidden_features = int(hidden_features * 2 / 3)
        hidden_features = math.ceil(hidden_features / align_to) * align_to
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias, **dd)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias, **dd)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias, **dd)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SelfBlock(nn.Module):
    """
    SelfBlock runs in sequence:
    - self attention layer with attention mask and RoPE
    - feed-forward layer

    Both with pre-norm, layer scale (optional), sample drop (optional), and KV cache (optional).

    When using sample drop, not all cases of `attn_mask` are guaranteed to work:
    - None -> ok
    - Bool/float tensor of shape `[N, N]` -> ok
    - Bool/float tensor of shape `[B, head, N, N]` -> ok, we explicitly handle this case by indexing on the batch dim
    - Something else, e.g. xformers or flex attention mask -> we don't handle this case, but:
      - if `attn_mask` is a `[N, N]` mask that is shared across all samples in the batch,
        it will probably work depending on the attention implementation
      - if `attn_mask` is different for each sample, it will definitely not work
    """

    def __init__(
        self,
        dim: int,
        *,
        attn_layer: Callable[..., nn.Module],
        ffn_layer: Callable[..., nn.Module],
        ls_init: float | None = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        sample_drop_rate: float = 0.0,  # From each batch, drop B * sample_drop_rate samples
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.norm1 = norm_layer(dim, **dd)
        self.attn = attn_layer(dim, **dd)
        self.ls1 = (
            LayerScale(dim, init=ls_init, **dd)
            if ls_init is not None
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = ffn_layer(dim, **dd)
        self.ls2 = (
            LayerScale(dim, init=ls_init, **dd)
            if ls_init is not None
            else nn.Identity()
        )

        self.sample_drop_rate = sample_drop_rate

    def forward(
        self,
        x: Tensor,  # [B, N, D]
        *,
        attn_mask: AttnMaskType = None,  # [N, N] or [B, head, N, N]
        rope: Tensor | None = None,  # [2, B, N', head, D_head] with N' <= N
        cache: None = None,  # Updated inplace
    ) -> Tensor:
        if self.training and self.sample_drop_rate > 0.0:
            if cache is not None:
                raise ValueError("KvCache is not supported with sample drop")
            return self._forward_sample_drop(x, attn_mask=attn_mask, rope=rope)
        return self._forward_simple(x, attn_mask=attn_mask, rope=rope, cache=cache)

    def _forward_simple(
        self,
        x: Tensor,  # [B, N, D]
        *,
        attn_mask: AttnMaskType,  # [N, N] or [B, head, N, N]
        rope: Tensor | None,  # [2, B, N', head, D_head] with N' <= N
        cache: None,  # Updated inplace
    ) -> Tensor:
        x_norm1 = self.norm1(x)
        residual1 = self.ls1(
            self.attn(x_norm1, attn_mask=attn_mask, rope=rope, cache=cache)
        )
        x = x + residual1

        x_norm2 = self.norm2(x)
        residual2 = self.ls2(self.mlp(x_norm2))
        x = x + residual2

        return x

    def _forward_sample_drop(
        self,
        x: Tensor,  # [B, N, D]
        *,
        attn_mask: AttnMaskType,  # [N, N] or [B, head, N, N]
        rope: Tensor | None,  # [2, B, N', head, D_head] with N' <= N
    ) -> Tensor:
        B, N, D = x.shape
        device = x.device
        sample_subset_size = max(round(B * (1 - self.sample_drop_rate)), 1)
        # scale factor > 1 because during training the network is stochastically shorter
        residual_scale_factor = B / sample_subset_size

        index1 = torch.randperm(B, device=device)[:sample_subset_size]
        x_norm1 = self.norm1(x[index1])
        if (
            isinstance(attn_mask, Tensor)
            and not isinstance(attn_mask, X.AttentionBias)
            and attn_mask.ndim == 4
            and attn_mask.shape[0] != 1
        ):
            attn_mask = attn_mask[index1]  # Different mask per-sample and per-head
        if rope is not None and rope.shape[1] != 1:
            rope = rope[:, index1]  # Per-sample rope
        residual1 = self.ls1(self.attn(x_norm1, attn_mask=attn_mask, rope=rope))
        x = torch.index_add(
            x, dim=0, source=residual_scale_factor * residual1, index=index1
        )

        index2 = torch.randperm(B, device=device)[:sample_subset_size]
        x_norm2 = self.norm2(x[index2])
        residual2 = self.ls2(self.mlp(x_norm2))
        x = torch.index_add(
            x, dim=0, source=residual_scale_factor * residual2, index=index2
        )

        return x


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


class QkvLinear(nn.Linear):
    """The first 1/3 of the bias, i.e. the query part, is always 0."""

    def forward(self, input: Tensor) -> Tensor:
        bias = self.bias
        if bias is not None:
            bias = bias.reshape(3, -1)
            bias = bias * bias.new_tensor([[0], [1], [1]])
            bias = bias.flatten()
        return F.linear(input, self.weight, bias)


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [..., D], eg [ x0  x1  x2  x3  x4  x5  x6  x7]
    # out: [..., D], eg [-x1  x0 -x3  x2  x5 -x4  x7  x6]
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)  # 2 * [..., D//2]
    return torch.stack([-b, a], dim=-1).flatten(-2)  # [..., D]


def rope_apply(x: Tensor, rope: Tensor) -> Tensor:
    # x:   [..., D], eg [  x0,   x1,   x2,   x3,   x4,   x5,   x6,   x7]
    # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2, cos3, cos3]
    # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2, sin3, sin3]
    # Use the dtype of rope, then cast back to the dtype of the input
    dtype = x.dtype
    cos, sin = rope.unbind(dim=0)  # 2 * [..., D]
    x = x.to(dtype=cos.dtype)
    x = (x * cos) + (rope_rotate_half(x) * sin)  # [..., D]
    x = x.to(dtype=dtype)
    return x


def rope_apply_with_prefix(
    x: Tensor,  # [..., L,  head, D]
    rope: Tensor,  # [2, ..., L', head, D] with L' <= L
) -> Tensor:
    L = x.shape[-3]
    prefix = L - rope.shape[-3]
    if prefix == 0:
        x = rope_apply(x, rope)  # [..., L, head, D]
    elif prefix > 0:
        x_prefix = x[..., :prefix, :, :]
        x_suffix = x[..., prefix:, :, :]  # [..., L', head, D]
        x_suffix = rope_apply(x_suffix, rope)
        x = torch.cat((x_prefix, x_suffix), dim=-3)  # [..., L, head, D]
    else:
        raise ValueError(f"Invalid negative {prefix=}")
    return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


from torch.testing._internal.distributed.fake_pg import FakeStore

world_size = 64

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", (world_size,), mesh_dim_names=("dp",)
)


device = torch.device("cuda")
cfg = Configuration(output_dir="")
cfg = Configuration(output_dir="", model_size="debug")
# cfg = Configuration(output_dir="", batch_size=16, model_size="large")
cfg = Configuration(output_dir="", batch_size=16, model_size="debug")


with torch.device("meta"):
    model = DinoVideoPretraining(cfg)
# model.to(device)


def input_fn(B=world_size):
    # B = cfg.batch_size * B
    local_batch = cfg.batch_size
    num_gpus = B
    B = local_batch * B
    T = cfg.num_frames

    # B = 2

    Hg, Wg = cfg.global_crop_size, cfg.global_crop_size
    Hl, Wl = cfg.local_crop_size, cfg.local_crop_size
    global_crop_num = cfg.global_crop_num
    local_crop_num = cfg.local_crop_num
    patch_size = cfg.patch_size

    h, w = Hg // patch_size, Wg // patch_size

    global_crops = torch.rand(B, global_crop_num, T, 3, Hg, Wg, device=device)
    local_crops = torch.rand(B, local_crop_num, T, 3, Hl, Wl, device=device)
    timestamps_global = torch.rand(B, global_crop_num, T, device=device)
    timestamps_local = torch.rand(B, local_crop_num, T, device=device)

    num_samples_masked = round(cfg.mask_sample_ratio * local_batch * global_crop_num)
    mask_bool = []
    num_non_zero = 0
    for r in np.linspace(*cfg.mask_patch_ratio, num_samples_masked).tolist():
        m = torch.randint(0, 2, (h, w), dtype=torch.bool)
        # print(round(r * h * w))
        num_non_zero += round(r * h * w)
        mask_bool.append(m)
    mask_bool = torch.stack(mask_bool)  # [num_samples_masked, h, w]
    mask_bool = mask_bool.repeat(num_gpus, 1, 1)
    mask_bool = F.pad(
        mask_bool, (0, 0, 0, 0, 0, B * global_crop_num - num_samples_masked)
    )
    mask_bool = mask_bool[
        torch.randperm(B * global_crop_num)
    ]  # [B * global_crop_num, h, w]
    mask_bool = mask_bool.reshape(
        B, global_crop_num, h, w
    )  # [B, global_crop_num, h, w]
    mask_bool = (
        mask_bool.unsqueeze(2).expand(-1, -1, T, -1, -1).clone()
    )  # [B, global_crop_num, num_frames, h, w]
    mask_bool = mask_bool.to(device)
    # from IPython import embed; embed(); exit()

    mask_nonzero = (
        mask_bool.flatten()
        .unflatten(0, (num_gpus, -1))
        .nonzero_static(size=num_non_zero * T * num_gpus)[:, -1]
        .unflatten(0, (num_gpus, -1))
    )
    mask_weight = 1 / mask_bool.flatten(2, 4).sum(-1).clamp_min(
        1.0
    )  # [B, global_crop_num]
    mask_weight = mask_weight[:, :, None, None, None].expand_as(mask_bool)
    mask_weight = (
        mask_weight.flatten().unflatten(0, (num_gpus, -1)).gather(1, mask_nonzero)
    )

    mask_weight /= num_samples_masked  # [num_masks] sums to 1.0
    return (
        global_crops,
        local_crops,
        timestamps_global,
        timestamps_local,
        mask_bool,
        mask_nonzero,
        mask_weight,
        0.1,
        False,
    )


# global_crops: [B, gc, T, rgb, H, W]
# local_crops: [B, lc, T, rgb, H', W']
# mask_bool: [B, gc, T, h, w]
# mask_nonzero: [B * gc * T * h * w * total_masked_ratio]
# mask_weight: [B * gc * T * h * w * total_masked_ratio]
# timestamps_global: [B, gc, T] ; this is used for rope
# timestamps_local: [B, lc, T] ; this is used for rope

# out = model(*input_fn())

import time

from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from autoparallel.api import AutoParallel

mp_policy = MixedPrecisionPolicy(
    param_dtype=DTYPES[cfg.param_dtype], reduce_dtype=DTYPES[cfg.reduce_dtype]
)

if False:
    # the UX of debugging shape errors during tracing is
    # not great, we need to improve it
    model = DinoVideoPretraining(cfg)
    model.to(device)
    x = input_fn(B=1)
    x = tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in x)
    o = model(*x)
    exit()

from autoparallel.auto_bucketing import (
    aten_autobucketing_config,
    aten_autobucketing_reordering_pass,
    configure_inductor_for_autobucketing,
)
from autoparallel.debug_helpers import make_custom_runtime_estimation

autobucketing_level = "aten"
if autobucketing_level == "aten":
    aten_autobucketing_config.custom_runtime_estimation = (
        make_custom_runtime_estimation(mesh)
    )
    # this is from the stacked pr in https://github.com/pytorch/pytorch/pull/163960
    torch._inductor.config.reorder_for_peak_memory = False
    torch._inductor.config.reorder_for_compute_comm_overlap = False
    aten_autobucketing_reordering_pass = partial(
        aten_autobucketing_reordering_pass,
        configs=aten_autobucketing_config,
    )
    torch._inductor.config.post_grad_custom_post_pass = (
        aten_autobucketing_reordering_pass
    )


with AutoParallel(
    model,
    input_fn,
    mesh,
    mp_policy,
    compile=False,
    repeated_subgraphs=False,  # True
) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    rep_sharding = (Replicate(),) * mesh.ndim

    autop.add_input_constraints([x_sharding] * 7 + [rep_sharding] * 2)
    autop.add_output_constraints([rep_sharding])

    t = time.time()
    sharding_placement = autop.optimize_placement(verbose=True)
    print(f"Took {time.time() - t:.2f} s")
    parallel_mod = autop.apply_placement(sharding_placement)
    # exit()

# run weight init on our sharded DTensor params
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

# now let's run it
x = input_fn(B=1)
# from IPython import embed; embed(); exit()
out = parallel_mod(*x)
# out[0].backward()
out.backward()
print("All good!")
