"""Algorithm understanding for NetPrism graphs (standard library only).

The layer answers three questions for a parsed model:

* **What is it?**  A family classifier (:func:`classify_families`) that scores
  every family from the knowledge base (``netprism.algorithm_kb``) using
  aliases, required/optional components and the graph's operator signature.
* **How is it built?**  Structural pattern detection over groups/nodes/edges
  (residual blocks, depthwise separable convs, transformer blocks, MoE, SSM,
  Swin windows, ...).  Every pattern carries evidence and a confidence.
* **What happens in forward?**  Per-node formulas (:func:`node_formula`) and a
  folded forward pseudocode with output shapes.

Everything is derived from the graph document.  No torch/onnx import happens,
nodes/edges/params/metrics are never modified and :func:`enrich_algorithms` is
idempotent (an existing ``model.meta.family`` is preserved unless ``force``).

The knowledge base is optional: when ``netprism.algorithm_kb`` is missing or
unreadable a small built-in fallback (the SPEC-ALGO families/components) keeps
the classifier working, so importing this module never fails.
"""

from __future__ import annotations

import heapq
import math
import re
from typing import Any, Callable, Iterable

try:  # pragma: no cover - schema is always present inside the package
    from .schema import KINDS as _SCHEMA_KINDS
except Exception:  # pragma: no cover - defensive for standalone use
    _SCHEMA_KINDS = (
        "input", "output", "conv", "deconv", "linear", "mlp", "norm",
        "activation", "pool", "dropout", "flatten", "reshape", "attention",
        "ffn", "embedding", "positional", "add", "concat", "upsample", "rnn",
        "softmax", "block", "container", "custom", "unknown",
    )

KINDS: tuple[str, ...] = tuple(_SCHEMA_KINDS)
_KIND_SET = frozenset(KINDS)

ALGORITHM_VERSION = 1
MAX_PATTERNS = 200
MAX_PSEUDOCODE_LINES = 200

#: SPEC-ALGO canonical families (kept first class for backward compatibility).
CANONICAL_FAMILIES: tuple[str, ...] = (
    "cnn",
    "vgg",
    "resnet",
    "densenet",
    "mobilenet",
    "mobilenet_v2",
    "inception",
    "unet",
    "transformer_encoder",
    "transformer_decoder",
    "vit",
    "mlp",
    "rnn",
    "hybrid",
    "unknown",
)

_FAMILY_LABELS: dict[str, str] = {
    "cnn": "CNN",
    "vgg": "VGG",
    "resnet": "ResNet",
    "densenet": "DenseNet",
    "mobilenet": "MobileNet",
    "mobilenet_v2": "MobileNetV2",
    "inception": "Inception",
    "unet": "UNet",
    "transformer_encoder": "Transformer Encoder",
    "transformer_decoder": "Transformer Decoder",
    "vit": "Vision Transformer",
    "mlp": "MLP",
    "rnn": "RNN",
    "hybrid": "Hybrid",
    "unknown": "Unknown",
}

# --------------------------------------------------------------------------- #
# pattern catalogue
# --------------------------------------------------------------------------- #
PATTERN_SPECS: dict[str, dict[str, Any]] = {
    "conv_bn_act": {
        "label": "Conv-BN-Act",
        "formula": "y = Act(BN(Conv(x)))",
        "tags": ["conv", "conv_bn_act", "normalization", "activation"],
    },
    "residual_block": {
        "label": "Residual Block",
        "formula": "y = F(x) + x",
        "tags": ["residual", "skip_connection"],
    },
    "bottleneck": {
        "label": "Bottleneck Block",
        "formula": "y = x + W3(W2(W1(x)))",
        "tags": ["bottleneck", "residual", "conv"],
    },
    "depthwise_separable": {
        "label": "Depthwise Separable Conv",
        "formula": "y = PW(DW(x))",
        "tags": ["depthwise", "depthwise_separable"],
    },
    "inverted_residual": {
        "label": "Inverted Residual Block",
        "formula": "y = x + Wp(DW(We(x)))",
        "tags": ["inverted_residual", "depthwise", "residual"],
    },
    "squeeze_excitation": {
        "label": "Squeeze-and-Excitation",
        "formula": "y = x · σ(W2·δ(W1·GAP(x)))",
        "tags": ["se", "channel_attention", "gating"],
    },
    "cbam": {
        "label": "CBAM Attention",
        "formula": "y = x · σ(MLP(AvgPool(x)) + MLP(MaxPool(x))) · σ(Conv7x7([avg; max]))",
        "tags": ["cbam", "channel_attention", "spatial_attention"],
    },
    "sk_attention": {
        "label": "SK Attention",
        "formula": "y = Σ_k a_k(x) · B_k(x)",
        "tags": ["sk", "selective_kernel", "attention"],
    },
    "inception": {
        "label": "Inception Module",
        "formula": "y = cat(B1(x), B2(x), ...)",
        "tags": ["inception", "multi_branch"],
    },
    "dense_connection": {
        "label": "Dense Connection",
        "formula": "x_l = H_l([x_0, ..., x_{l-1}])",
        "tags": ["dense", "concat"],
    },
    "unet_skip": {
        "label": "UNet Skip Connection",
        "formula": "y = Dec(cat(x_enc, up(x)))",
        "tags": ["unet", "skip_connection", "concat"],
    },
    "unet_plus_plus": {
        "label": "UNet++ Dense Skip",
        "formula": "x^{i,j} = H(cat(x^{i,j-1}, x^{i+1,j-1}, ...))",
        "tags": ["unet_plus_plus", "dense", "skip_connection"],
    },
    "fpn": {
        "label": "FPN Lateral Fusion",
        "formula": "P_l = Conv_l(x_l) + Up(P_{l+1})",
        "tags": ["fpn", "multi_scale", "upsample"],
    },
    "panet": {
        "label": "PANet Bidirectional Fusion",
        "formula": "P_l = Conv_l(x_l) + Up(P_{l+1}) + Down(P_{l-1})",
        "tags": ["panet", "multi_scale", "bidirectional"],
    },
    "transformer_block": {
        "label": "Transformer Block",
        "formula": "x = x + MHA(LN(x)); x = x + FFN(LN(x))",
        "tags": ["transformer", "attention", "residual"],
    },
    "self_attention": {
        "label": "Self Attention",
        "formula": "softmax(Q·Kᵀ/√d)·V, Q/K/V 同源",
        "tags": ["attention", "self_attention"],
    },
    "cross_attention": {
        "label": "Cross Attention",
        "formula": "softmax(Q·Kᵀ/√d)·V, K/V 来自另一分支",
        "tags": ["attention", "cross_attention"],
    },
    "ffn": {
        "label": "Feed-Forward Network",
        "formula": "y = W2·Act(W1·x)",
        "tags": ["ffn", "expansion"],
    },
    "gated_mlp": {
        "label": "Gated MLP",
        "formula": "y = W2(Act(W1·x) ⊙ σ(W3·x))",
        "tags": ["gated_mlp", "swiglu", "gating"],
    },
    "pre_norm": {
        "label": "Pre-Norm",
        "formula": "x = x + F(LN(x))",
        "tags": ["norm", "pre_norm"],
    },
    "post_norm": {
        "label": "Post-Norm",
        "formula": "x = LN(x + F(x))",
        "tags": ["norm", "post_norm"],
    },
    "rnn_block": {
        "label": "RNN Block",
        "formula": "h_t = RNN(x_t, h_{t-1})",
        "tags": ["rnn", "sequence"],
    },
    "weight_tying": {
        "label": "Weight Tying",
        "formula": "共享权重复用（W_head = W_embed）",
        "tags": ["weight_tying", "parameter_sharing"],
    },
    "moe": {
        "label": "Mixture of Experts",
        "formula": "y = Σ_i g_i(x)·E_i(x), g = softmax(W_g·x)",
        "tags": ["moe", "expert", "gating"],
    },
    "ssm": {
        "label": "State-Space Block (Mamba)",
        "formula": "h_t = Ā·h_{t-1} + B̄·x_t, y_t = C·h_t",
        "tags": ["ssm", "mamba", "selective_scan"],
    },
    "rwkv": {
        "label": "RWKV Time-Mix",
        "formula": "y = σ(r)·(W_v·x) + (1-σ(r))·(W_k·x)",
        "tags": ["rwkv", "time_mix", "linear_attention"],
    },
    "gnn_message_passing": {
        "label": "GNN Message Passing",
        "formula": "h_v' = φ(h_v, ⊕_{u∈N(v)} ψ(h_u, h_v))",
        "tags": ["gnn", "message_passing", "graph"],
    },
    "point_mlp": {
        "label": "PointNet MLP",
        "formula": "y = max_p(MLP(x_p))",
        "tags": ["point_mlp", "pointnet", "permutation_invariant"],
    },
    "nerf_mlp": {
        "label": "NeRF MLP",
        "formula": "y = MLP(γ(x), γ(d)) → (σ, c)",
        "tags": ["nerf", "positional_encoding", "implicit_field"],
    },
    "vae_reparam": {
        "label": "VAE Reparameterization",
        "formula": "z = μ + σ ⊙ ε, σ = exp(0.5·logσ²)",
        "tags": ["vae", "reparameterization", "latent"],
    },
    "diffusion_unet": {
        "label": "Diffusion UNet",
        "formula": "ε_θ(x_t, t) = UNet(cat(x_t, time_embed(t)))",
        "tags": ["diffusion", "unet", "time_embedding"],
    },
    "swin_attention": {
        "label": "Swin Window Attention",
        "formula": "y = W-MSA(reshape(x, window)) + ShiftedWindow",
        "tags": ["swin", "window_attention", "shift"],
    },
    "rope": {
        "label": "RoPE",
        "formula": "q_rot = q·cos(θ) + rotate_half(q)·sin(θ)",
        "tags": ["rope", "rotary", "positional_encoding"],
    },
    "rmsnorm": {
        "label": "RMSNorm",
        "formula": "y = γ·x/√(mean(x²)+ε)",
        "tags": ["rmsnorm", "normalization"],
    },
    "lora": {
        "label": "LoRA Adapter",
        "formula": "y = W·x + (α/r)·B·A·x",
        "tags": ["lora", "low_rank", "adapter"],
    },
    "patch_embedding": {
        "label": "Patch Embedding",
        "formula": "y = Conv(k=patch, s=patch)(x)",
        "tags": ["patch_embedding", "vit"],
    },
    "positional_encoding": {
        "label": "Positional Encoding",
        "formula": "y = x + PE(pos, d)",
        "tags": ["positional_encoding", "sequence"],
    },
    "global_pool": {
        "label": "Global Pool",
        "formula": "y = mean_{h,w}(x)",
        "tags": ["global_pool", "pool"],
    },
    "classifier_head": {
        "label": "Classifier Head",
        "formula": "logits = W·x + b",
        "tags": ["classifier_head", "linear"],
    },
}

#: pattern kind -> extra structural signal emitted for the classifier.
_PATTERN_SIGNAL: dict[str, str] = {
    "residual_block": "residual",
    "depthwise_separable": "depthwise_separable",
    "inverted_residual": "inverted_residual",
    "squeeze_excitation": "se",
    "dense_connection": "dense",
    "rnn_block": "rnn_block",
    "gnn_message_passing": "gnn",
    "point_mlp": "point_mlp",
    "nerf_mlp": "nerf",
    "vae_reparam": "vae",
    "diffusion_unet": "diffusion",
    "swin_attention": "swin",
    "positional_encoding": "positional",
    "patch_embedding": "patch_embed",
    "sk_attention": "sk",
}

#: group level pattern priority used for ``group.meta.pattern``.
_GROUP_PATTERN_PRIORITY: dict[str, int] = {
    "diffusion_unet": 110,
    "moe": 108,
    "transformer_block": 105,
    "unet_plus_plus": 100,
    "swin_attention": 98,
    "rwkv": 96,
    "ssm": 95,
    "gnn_message_passing": 94,
    "point_mlp": 93,
    "nerf_mlp": 92,
    "vae_reparam": 91,
    "unet_skip": 90,
    "inverted_residual": 85,
    "bottleneck": 84,
    "residual_block": 80,
    "dense_connection": 78,
    "inception": 76,
    "fpn": 74,
    "panet": 73,
    "squeeze_excitation": 70,
    "cbam": 69,
    "sk_attention": 68,
    "gated_mlp": 66,
    "ffn": 64,
    "self_attention": 62,
    "cross_attention": 61,
    "rnn_block": 60,
    "depthwise_separable": 55,
    "conv_bn_act": 50,
    "pre_norm": 45,
    "post_norm": 44,
    "weight_tying": 40,
    "patch_embedding": 35,
    "rope": 34,
    "rmsnorm": 33,
    "lora": 32,
    "classifier_head": 30,
    "global_pool": 25,
    "positional_encoding": 20,
}

#: pattern kind -> compact block call used by the pseudocode.
_BLOCK_NAMES: dict[str, str] = {
    "residual_block": "ResidualBlock",
    "bottleneck": "Bottleneck",
    "conv_bn_act": "ConvBlock",
    "depthwise_separable": "DepthwiseSeparable",
    "inverted_residual": "InvertedResidual",
    "squeeze_excitation": "SEBlock",
    "cbam": "CBAM",
    "sk_attention": "SKAttention",
    "inception": "Inception",
    "dense_connection": "DenseBlock",
    "unet_skip": "UNetSkip",
    "unet_plus_plus": "UNetPlusPlus",
    "fpn": "FPN",
    "panet": "PANet",
    "transformer_block": "TransformerBlock",
    "attention_block": "AttentionBlock",
    "self_attention": "SelfAttention",
    "cross_attention": "CrossAttention",
    "ffn": "FeedForward",
    "gated_mlp": "GatedMLP",
    "rnn_block": "RNNBlock",
    "moe": "MoE",
    "ssm": "MambaBlock",
    "rwkv": "RWKVBlock",
    "gnn_message_passing": "GNNLayer",
    "point_mlp": "PointMLP",
    "nerf_mlp": "NeRFMLP",
    "vae_reparam": "VAEReparam",
    "diffusion_unet": "DiffusionUNet",
    "swin_attention": "SwinBlock",
    "rmsnorm": "RMSNorm",
    "lora": "LoRALinear",
}

_FOLD_KINDS = frozenset(
    {
        "conv_block",
        "residual_block",
        "transformer_block",
        "attention_block",
        "ffn_block",
        "unet_block",
    }
)

#: Patterns that describe a whole block (foldable) rather than a node-level
#: annotation such as ``classifier_head`` / ``global_pool``.
_FOLDABLE_PATTERN_KINDS = frozenset(
    {
        "residual_block",
        "bottleneck",
        "conv_bn_act",
        "depthwise_separable",
        "inverted_residual",
        "squeeze_excitation",
        "cbam",
        "sk_attention",
        "inception",
        "dense_connection",
        "transformer_block",
        "self_attention",
        "cross_attention",
        "ffn",
        "gated_mlp",
        "rnn_block",
        "moe",
        "ssm",
        "rwkv",
        "gnn_message_passing",
        "point_mlp",
        "nerf_mlp",
        "vae_reparam",
        "swin_attention",
    }
)

# --------------------------------------------------------------------------- #
# knowledge base loading
# --------------------------------------------------------------------------- #
_KB_CACHE: dict[str, Any] = {"resolved": False, "data": None}


def _external_knowledge() -> dict[str, Any] | None:
    """Load the optional external knowledge base, returning ``None`` on failure."""
    try:
        from .algorithm_kb import load_knowledge  # type: ignore

        data = load_knowledge()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    families = data.get("families")
    if not isinstance(families, list) or not families:
        return None
    return data


def _component(
    cid: str,
    label: str,
    formula: str,
    description: str,
    required_ops: Iterable[str],
    tags: Iterable[str],
) -> dict[str, Any]:
    return {
        "id": cid,
        "label": label,
        "formula": formula,
        "description": description,
        "required_ops": [op for op in required_ops if op in _KIND_SET],
        "tags": list(tags),
    }


def _builtin_knowledge() -> dict[str, Any]:
    """Minimal fallback KB used when ``netprism.algorithm_kb`` is unavailable."""
    components = [
        _component("conv", "Convolution", "y = Conv(x)", "空间卷积层", ["conv"], ["conv"]),
        _component("conv_bn_act", "Conv-BN-Act", "y = Act(BN(Conv(x)))", "卷积+归一化+激活链", ["conv", "norm", "activation"], ["conv", "norm"]),
        _component("norm", "Normalization", "y = Norm(x)", "归一化层", ["norm"], ["norm"]),
        _component("pool", "Pooling", "y = Pool(x)", "池化下采样", ["pool"], ["pool"]),
        _component("global_pool", "Global Pool", "y = GAP(x)", "全局池化", ["pool"], ["pool"]),
        _component("classifier_head", "Classifier Head", "logits = W·x", "末端分类头", ["linear"], ["linear"]),
        _component("residual", "Residual", "y = F(x) + x", "残差/跨层连接", ["add"], ["residual"]),
        _component("bottleneck", "Bottleneck", "y = x + W3(W2(W1(x)))", "1x1-3x3-1x1 沙漏块", ["conv"], ["residual"]),
        _component("depthwise_separable", "Depthwise Separable", "y = PW(DW(x))", "深度可分离卷积", ["conv"], ["depthwise"]),
        _component("inverted_residual", "Inverted Residual", "y = x + Wp(DW(We(x)))", "MobileNetV2 倒残差", ["conv", "add"], ["depthwise"]),
        _component("se", "Squeeze-and-Excitation", "y = x · σ(W2δ(W1GAP(x)))", "通道注意力", ["pool", "linear"], ["attention"]),
        _component("inception", "Inception", "y = cat(B1(x), B2(x), ...)", "多分支多卷积核", ["conv", "concat"], ["multi_branch"]),
        _component("dense", "Dense Connection", "x_l = H_l([x_0, ..., x_{l-1}])", "DenseNet 密集连接", ["concat"], ["dense"]),
        _component("unet_skip", "UNet Skip", "y = Dec(cat(x_enc, up(x)))", "编码器-解码器跳连", ["concat", "upsample"], ["unet"]),
        _component("transformer_block", "Transformer Block", "x = x + MHA(LN(x)); x = x + FFN(LN(x))", "注意力+FFN 残差块", ["attention", "add"], ["transformer"]),
        _component("self_attention", "Self Attention", "softmax(QKᵀ/√d)V", "Q/K/V 同源注意力", ["attention"], ["attention"]),
        _component("cross_attention", "Cross Attention", "softmax(QKᵀ/√d)V", "Q 与 K/V 异源注意力", ["attention"], ["attention"]),
        _component("ffn", "Feed-Forward Network", "y = W2·Act(W1·x)", "扩张-激活-收缩", ["linear", "activation"], ["ffn"]),
        _component("gated_mlp", "Gated MLP", "y = W2(Act(W1x) ⊙ σ(W3x))", "两条线性分支门控相乘", ["linear", "activation"], ["gated_mlp"]),
        _component("pre_norm", "Pre-Norm", "x = x + F(LN(x))", "归一化在子层之前", ["norm"], ["norm"]),
        _component("post_norm", "Post-Norm", "x = LN(x + F(x))", "归一化在子层之后", ["norm"], ["norm"]),
        _component("rnn", "RNN", "h_t = RNN(x_t, h_{t-1})", "循环网络", ["rnn"], ["rnn"]),
        _component("mlp", "MLP", "y = W_n·Act(...W1x)", "多层感知机", ["linear", "activation"], ["mlp"]),
        _component("linear", "Linear", "y = W·x", "线性层", ["linear"], ["linear"]),
        _component("embedding", "Embedding", "y = E[token]", "查表嵌入", ["embedding"], ["embedding"]),
        _component("positional", "Positional", "y = x + PE", "位置编码", ["positional", "add"], ["positional"]),
        _component("upsample", "Upsample", "y = resize(x)", "上采样/转置卷积", ["upsample", "deconv"], ["upsample"]),
        _component("weight_tying", "Weight Tying", "共享权重", "权重共享", ["linear", "embedding"], ["weight_tying"]),
        _component("patch_embedding", "Patch Embedding", "y = Conv(k=patch, s=patch)(x)", "ViT 分块嵌入", ["conv"], ["patch"]),
        _component("skip", "Skip Connection", "y = F(x) + x", "跨层连接", ["add", "concat"], ["skip"]),
    ]
    families = [
        {
            "id": "cnn",
            "label": "CNN",
            "domain": "vision",
            "aliases": ["cnn", "convnet", "lenet", "alexnet"],
            "summary": "通用卷积网络",
            "components": ["conv", "conv_bn_act", "norm", "pool", "global_pool", "classifier_head"],
            "required_patterns": ["conv"],
            "optional_patterns": ["conv_bn_act", "pool", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "pool", "linear"],
            "formula": "y = Head(GAP(ConvBlocks(x)))",
            "references": [],
        },
        {
            "id": "vgg",
            "label": "VGG",
            "domain": "vision",
            "aliases": ["vgg", "vgg11", "vgg16", "vgg19", "vggnet"],
            "summary": "重复 3x3 卷积堆叠",
            "components": ["conv", "conv_bn_act", "pool", "global_pool", "classifier_head"],
            "required_patterns": ["conv", "conv_bn_act"],
            "optional_patterns": ["pool", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "pool", "linear"],
            "formula": "y = Head(GAP(3x3 ConvStack(x)))",
            "references": [],
        },
        {
            "id": "resnet",
            "label": "ResNet",
            "domain": "vision",
            "aliases": ["resnet", "res", "basicblock", "resnet18", "resnet34", "resnet50"],
            "summary": "残差网络",
            "components": ["conv", "conv_bn_act", "residual", "bottleneck", "global_pool", "classifier_head"],
            "required_patterns": ["conv", "residual"],
            "optional_patterns": ["conv_bn_act", "bottleneck", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "add", "pool"],
            "formula": "y = Head(GAP(ResidualBlocks(x)))",
            "references": [],
        },
        {
            "id": "densenet",
            "label": "DenseNet",
            "domain": "vision",
            "aliases": ["densenet", "dense"],
            "summary": "密集连接网络",
            "components": ["conv", "conv_bn_act", "dense", "global_pool", "classifier_head"],
            "required_patterns": ["conv", "dense"],
            "optional_patterns": ["conv_bn_act", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "concat", "pool"],
            "formula": "x_l = H_l([x_0, ..., x_{l-1}])",
            "references": [],
        },
        {
            "id": "mobilenet",
            "label": "MobileNet",
            "domain": "vision",
            "aliases": ["mobilenet", "mobilenetv1"],
            "summary": "深度可分离卷积网络",
            "components": ["conv", "depthwise_separable", "conv_bn_act", "global_pool", "classifier_head"],
            "required_patterns": ["conv", "depthwise_separable"],
            "optional_patterns": ["conv_bn_act", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "add", "pool"],
            "formula": "y = Head(GAP(PW(DW(x))...))",
            "references": [],
        },
        {
            "id": "mobilenet_v2",
            "label": "MobileNetV2",
            "domain": "vision",
            "aliases": ["mobilenetv2", "mobilenet_v2", "mbv2", "invertedresidual"],
            "summary": "倒残差+线性瓶颈",
            "components": ["conv", "inverted_residual", "depthwise_separable", "conv_bn_act", "global_pool", "classifier_head"],
            "required_patterns": ["conv", "inverted_residual"],
            "optional_patterns": ["depthwise_separable", "conv_bn_act", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "add", "pool"],
            "formula": "y = Head(GAP(InvertedResidualBlock...))",
            "references": [],
        },
        {
            "id": "inception",
            "label": "Inception",
            "domain": "vision",
            "aliases": ["inception", "googlenet", "xception"],
            "summary": "多分支多尺度卷积",
            "components": ["conv", "inception", "conv_bn_act", "pool", "global_pool", "classifier_head"],
            "required_patterns": ["inception"],
            "optional_patterns": ["conv", "conv_bn_act", "pool", "global_pool", "classifier_head"],
            "signature_ops": ["conv", "norm", "activation", "concat", "pool"],
            "formula": "y = cat(B1(x), B2(x), ...)",
            "references": [],
        },
        {
            "id": "unet",
            "label": "UNet",
            "domain": "vision",
            "aliases": ["unet", "u-net", "unetmini", "encoderdecoder"],
            "summary": "编码器-解码器跳连分割网络",
            "components": ["conv", "conv_bn_act", "unet_skip", "upsample"],
            "required_patterns": ["conv", "unet_skip", "upsample"],
            "optional_patterns": ["conv_bn_act"],
            "signature_ops": ["conv", "norm", "activation", "pool", "upsample", "concat"],
            "formula": "y = Decoder(Skip(Encoder(x)))",
            "references": [],
        },
        {
            "id": "transformer_encoder",
            "label": "Transformer Encoder",
            "domain": "sequence",
            "aliases": ["transformerencoder", "encoder", "bert", "transformer"],
            "summary": "Transformer 编码器",
            "components": ["transformer_block", "self_attention", "ffn", "embedding", "pre_norm", "post_norm"],
            "required_patterns": ["transformer_block", "self_attention", "ffn"],
            "optional_patterns": ["embedding", "pre_norm", "post_norm", "pool", "classifier_head"],
            "signature_ops": ["embedding", "attention", "linear", "norm", "add", "activation"],
            "formula": "x = TransformerEncoder(x)",
            "references": [],
        },
        {
            "id": "transformer_decoder",
            "label": "Transformer Decoder",
            "domain": "sequence",
            "aliases": ["transformerdecoder", "decoder", "gpt", "llm", "decoderonly"],
            "summary": "Transformer 解码器 / GPT",
            "components": ["transformer_block", "self_attention", "ffn", "embedding", "pre_norm", "post_norm", "weight_tying"],
            "required_patterns": ["transformer_block", "self_attention", "ffn"],
            "optional_patterns": ["embedding", "pre_norm", "post_norm", "weight_tying", "positional"],
            "signature_ops": ["embedding", "attention", "linear", "norm", "add", "activation"],
            "formula": "x = TransformerDecoder(x)",
            "references": [],
        },
        {
            "id": "vit",
            "label": "Vision Transformer",
            "domain": "vision",
            "aliases": ["vit", "visiontransformer", "deit", "swin"],
            "summary": "图像分块 + Transformer",
            "components": ["patch_embedding", "transformer_block", "self_attention", "ffn", "embedding"],
            "required_patterns": ["patch_embedding", "transformer_block", "self_attention"],
            "optional_patterns": ["ffn", "positional", "classifier_head"],
            "signature_ops": ["conv", "attention", "linear", "norm", "add", "activation"],
            "formula": "y = Head(Transformer(PatchEmbed(x)))",
            "references": [],
        },
        {
            "id": "mlp",
            "label": "MLP",
            "domain": "general",
            "aliases": ["mlp", "perceptron", "feedforward", "mixer"],
            "summary": "多层感知机",
            "components": ["mlp", "linear", "ffn", "gated_mlp"],
            "required_patterns": ["linear", "mlp"],
            "optional_patterns": ["activation", "gated_mlp", "dropout"],
            "signature_ops": ["linear", "activation", "dropout"],
            "formula": "y = W_n·Act(...W1·x)",
            "references": [],
        },
        {
            "id": "rnn",
            "label": "RNN",
            "domain": "sequence",
            "aliases": ["rnn", "lstm", "gru", "recurrent"],
            "summary": "循环神经网络",
            "components": ["rnn"],
            "required_patterns": ["rnn"],
            "optional_patterns": ["linear", "mlp"],
            "signature_ops": ["rnn", "linear", "activation"],
            "formula": "h_t = RNN(x_t, h_{t-1})",
            "references": [],
        },
        {
            "id": "hybrid",
            "label": "Hybrid",
            "domain": "general",
            "aliases": ["hybrid", "convtransformer", "conformer"],
            "summary": "卷积 + Transformer 混合",
            "components": ["conv", "conv_bn_act", "transformer_block", "self_attention"],
            "required_patterns": ["conv", "transformer_block"],
            "optional_patterns": ["conv_bn_act", "self_attention", "ffn"],
            "signature_ops": ["conv", "attention", "linear", "norm", "add"],
            "formula": "y = Transformer(ConvStem(x))",
            "references": [],
        },
    ]
    return {"version": "fallback", "families": families, "components": components}


def _knowledge() -> dict[str, Any]:
    """Return the knowledge base (external when available, cached)."""
    if _KB_CACHE.get("resolved"):
        data = _KB_CACHE.get("data")
        if isinstance(data, dict):
            return data
    data = _external_knowledge()
    if not isinstance(data, dict):
        data = _builtin_knowledge()
    _KB_CACHE["data"] = data
    _KB_CACHE["resolved"] = True
    return data


# --------------------------------------------------------------------------- #
# small value helpers
# --------------------------------------------------------------------------- #
def _node_kind(node: Any) -> str:
    if not isinstance(node, dict):
        return "unknown"
    kind = node.get("kind")
    if isinstance(kind, str) and kind:
        return kind.lower()
    return "unknown"


def _op_short(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    op = node.get("op")
    if not isinstance(op, str):
        return ""
    return op.rsplit(".", 1)[-1]


def _params(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    params = node.get("params")
    return params if isinstance(params, dict) else {}


def _param(params: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
    return None


def _as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                out.append(int(item))
        return out
    return []


def _shapes(value: Any) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value):
        return [[int(dim) for dim in value]]
    shapes: list[list[int]] = []
    for shape in value:
        if not isinstance(shape, (list, tuple)) or not shape:
            continue
        if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
            shapes.append([int(dim) for dim in shape])
    return shapes


def _first_shape(node: Any, key: str = "outShapes") -> list[int] | None:
    if not isinstance(node, dict):
        return None
    shapes = _shapes(node.get(key))
    return shapes[0] if shapes else None


def _rank(node: Any) -> int | None:
    shape = _first_shape(node) or _first_shape(node, "inShapes")
    return len(shape) if shape else None


def _axes_for_rank(rank: int | None, *, sequence: bool = False) -> str | None:
    if rank == 4:
        return "NCHW"
    if rank == 5:
        return "NCDHW"
    if rank == 3:
        return "NLC (B,L,E)" if sequence else "NCW"
    if rank == 2:
        return "N×F"
    if rank == 1:
        return "C"
    return None


def _kernel_list(params: dict[str, Any]) -> list[int]:
    return _as_int_list(_param(params, "kernelSize", "kernel_size", "kernel"))


def _kernel_text(params: dict[str, Any]) -> str:
    kernel = _kernel_list(params)
    if not kernel:
        return "?"
    return "x".join(str(max(1, dim)) for dim in kernel)


def _int_param(params: dict[str, Any], *names: str, default: int = 0) -> int:
    value = _as_int(_param(params, *names))
    return default if value is None else value


def _is_pointwise(node: Any) -> bool:
    if _node_kind(node) != "conv":
        return False
    kernel = _kernel_list(_params(node))
    return bool(kernel) and all(dim == 1 for dim in kernel)


def _is_depthwise(node: Any) -> bool:
    if _node_kind(node) != "conv":
        return False
    params = _params(node)
    groups = _int_param(params, "groups", default=1) or 1
    cin = _int_param(params, "inChannels", "in_channels")
    if groups <= 1 or cin <= 1:
        return False
    return groups == cin


def _is_global_pool(ctx: "_Ctx", node: Any) -> bool:
    if _node_kind(node) != "pool":
        return False
    params = _params(node)
    op = _op_short(node).lower()
    tags = node.get("tags") if isinstance(node, dict) else None
    if isinstance(tags, list) and "global" in [str(t).lower() for t in tags]:
        return True
    output_size = _as_int_list(_param(params, "outputSize", "output_size"))
    if output_size and all(dim == 1 for dim in output_size):
        return True
    out = _first_shape(node)
    if out and len(out) >= 3 and all(dim in (1, -1) for dim in out[2:]):
        return True
    if op in ("mean", "sum") and _param(params, "dim") is not None:
        return True
    if "adaptive" in op and out and len(out) == 2:
        return True
    return False


def _is_global_max_pool(ctx: "_Ctx", node: Any) -> bool:
    if not _is_global_pool(ctx, node):
        return False
    op = _op_short(node).lower()
    params = _params(node)
    ptype = str(_param(params, "type") or "").lower()
    return "max" in op or ptype == "max"


def _is_mul(node: Any) -> bool:
    op = _op_short(node).lower()
    if op in ("mul", "mul_", "__mul__"):
        return True
    kind = _node_kind(node)
    name = str(node.get("name") or "").lower() if isinstance(node, dict) else ""
    return kind in ("custom", "add") and "mul" in name


def _is_sigmoid(node: Any) -> bool:
    return _node_kind(node) == "activation" and _op_short(node).lower() in ("sigmoid", "hardsigmoid", "logsigmoi")


def _iter_activation(node: Any) -> str:
    return _op_short(node).lower()


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^0-9a-z]+", "", value.lower())


_ALIAS_STOPWORDS = {
    "net",
    "network",
    "model",
    "block",
    "layer",
    "deep",
    "neural",
    "module",
    "arch",
    "architecture",
    "classifier",
    "encoder",
    "decoder",
    "base",
}


def _name_candidates(model: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("name", "className"):
        norm = _norm_text(model.get(key))
        if norm:
            names.append(norm)
    source = model.get("source")
    if isinstance(source, dict):
        rel_path = source.get("relPath") or source.get("file")
        if isinstance(rel_path, str):
            stem = rel_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            norm = _norm_text(stem)
            if norm:
                names.append(norm)
    return names


def _name_tokens(nodes: Iterable[dict[str, Any]], model: dict[str, Any]) -> set[str]:
    texts: list[str] = []
    for key in ("name", "className"):
        value = model.get(key)
        if isinstance(value, str):
            texts.append(value)
    source = model.get("source")
    if isinstance(source, dict):
        value = source.get("relPath") or source.get("file")
        if isinstance(value, str):
            texts.append(value)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key in ("name", "modulePath", "op"):
            value = node.get(key)
            if isinstance(value, str):
                texts.append(value)
    tokens: set[str] = set()
    for text in texts:
        for part in re.split(r"[^0-9A-Za-z]+", text):
            if not part:
                continue
            for token in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part):
                tokens.add(token.lower())
            tokens.add(part.lower())
    return tokens


# --------------------------------------------------------------------------- #
# graph context
# --------------------------------------------------------------------------- #
class _Ctx:
    """Read-only indexed view over a graph document."""

    def __init__(self, graph: Any) -> None:
        self.graph: dict[str, Any] = graph if isinstance(graph, dict) else {}
        model = self.graph.get("model")
        self.model: dict[str, Any] = model if isinstance(model, dict) else {}

        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_ids: list[str] = []
        raw_nodes = self.graph.get("nodes")
        if isinstance(raw_nodes, list):
            for index, node in enumerate(raw_nodes):
                if not isinstance(node, dict):
                    continue
                node_id = node.get("id")
                if not isinstance(node_id, str) or not node_id:
                    node_id = f"n{index}"
                    node = dict(node)
                    node["id"] = node_id
                if node_id in self.nodes:
                    continue
                self.nodes[node_id] = node
                self.node_ids.append(node_id)

        self.edges: list[dict[str, Any]] = []
        self.preds: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        self.succs: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        self.pred_edges: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in self.nodes}
        raw_edges = self.graph.get("edges")
        seen_pairs: set[tuple[str, str]] = set()
        if isinstance(raw_edges, list):
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                source, target = edge.get("source"), edge.get("target")
                if source not in self.nodes or target not in self.nodes or source == target:
                    continue
                pair = (source, target)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                self.edges.append(edge)
                self.preds[target].append(source)
                self.succs[source].append(target)
                self.pred_edges[target].append(edge)

        self.order = self._topo_order()
        self.pos = {node_id: index for index, node_id in enumerate(self.order)}

        self.kind_counts: dict[str, int] = {}
        for node_id in self.node_ids:
            kind = _node_kind(self.nodes[node_id])
            self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1

        # groups ----------------------------------------------------------
        self.groups: list[dict[str, Any]] = []
        self.group_by_id: dict[str, dict[str, Any]] = {}
        raw_groups = self.graph.get("groups")
        if isinstance(raw_groups, list):
            for group in raw_groups:
                if not isinstance(group, dict):
                    continue
                group_id = group.get("id")
                if not isinstance(group_id, str) or not group_id or group_id in self.group_by_id:
                    continue
                self.groups.append(group)
                self.group_by_id[group_id] = group

        self.group_direct: dict[str, list[str]] = {}
        for group in self.groups:
            group_id = group["id"]
            direct: list[str] = []
            seen: set[str] = set()
            for node_id in group.get("nodeIds") or []:
                if isinstance(node_id, str) and node_id in self.nodes and node_id not in seen:
                    seen.add(node_id)
                    direct.append(node_id)
            self.group_direct[group_id] = direct

        self.group_children: dict[str, list[str]] = {group["id"]: [] for group in self.groups}
        self.group_parent: dict[str, str] = {}
        for group in self.groups:
            group_id = group["id"]
            parent = group.get("parentId")
            if isinstance(parent, str) and parent in self.group_by_id and parent != group_id:
                self.group_parent[group_id] = parent
            seen_children: set[str] = set()
            for child in group.get("childGroupIds") or []:
                if (
                    isinstance(child, str)
                    and child in self.group_by_id
                    and child != group_id
                    and child not in seen_children
                ):
                    seen_children.add(child)
                    self.group_children[group_id].append(child)
        for group_id, parent in list(self.group_parent.items()):
            if group_id not in self.group_children.get(parent, []):
                self.group_children.setdefault(parent, []).append(group_id)

        roots = [group["id"] for group in self.groups if group["id"] not in self.group_parent]
        roots.sort(key=lambda gid: (int(self.group_by_id[gid].get("depth") or 0), str(self.group_by_id[gid].get("name") or gid)))
        self.root_ids: list[str] = roots
        self.root_group_id: str | None = roots[0] if roots else None

        self.node_group: dict[str, str] = {}
        for group in self.groups:
            for node_id in self.group_direct.get(group["id"], []):
                self.node_group.setdefault(node_id, group["id"])
        for node_id in self.node_ids:
            raw_group = self.nodes[node_id].get("groupId")
            if isinstance(raw_group, str) and raw_group in self.group_by_id:
                self.node_group.setdefault(node_id, raw_group)

        self._subtree_cache: dict[str, list[str]] = {}

    # -- ordering ------------------------------------------------------
    @staticmethod
    def _sort_key(node_id: str) -> tuple[int, str]:
        if node_id.startswith("n") and node_id[1:].isdigit():
            return (int(node_id[1:]), node_id)
        return (10**9, node_id)

    def _topo_order(self) -> list[str]:
        indegree = {node_id: 0 for node_id in self.nodes}
        for target, sources in self.preds.items():
            indegree[target] = len(sources)
        heap: list[tuple[tuple[int, str], str]] = []
        for node_id, degree in indegree.items():
            if degree == 0:
                heapq.heappush(heap, (self._sort_key(node_id), node_id))
        order: list[str] = []
        while heap:
            _, node_id = heapq.heappop(heap)
            order.append(node_id)
            for target in self.succs[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(heap, (self._sort_key(target), target))
        placed = set(order)
        for node_id in sorted((nid for nid in self.nodes if nid not in placed), key=self._sort_key):
            order.append(node_id)
        return order

    # -- group helpers -------------------------------------------------
    def subtree_nodes(self, group_id: str) -> list[str]:
        cached = self._subtree_cache.get(group_id)
        if cached is not None:
            return cached
        result: list[str] = []
        seen: set[str] = set()
        queue = [group_id]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for node_id in self.group_direct.get(current, []):
                if node_id not in seen:
                    seen.add(node_id)
                    result.append(node_id)
            for child in self.group_children.get(current, []):
                if child not in visited:
                    queue.append(child)
        result.sort(key=lambda nid: self.pos.get(nid, 10**9))
        self._subtree_cache[group_id] = result
        return result

    def group_chain(self, group_id: str) -> list[str]:
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = group_id
        while current is not None and current in self.group_by_id and current not in seen:
            seen.add(current)
            chain.append(current)
            current = self.group_parent.get(current)
        return chain

    def groups_related(self, first: str, second: str) -> bool:
        if first == second:
            return True
        return first in self.group_chain(second) or second in self.group_chain(first)

    def lca_group(self, group_ids: Iterable[str]) -> str | None:
        ids = [gid for gid in group_ids if gid in self.group_by_id]
        if not ids:
            return None
        common = set(self.group_chain(ids[0]))
        for gid in ids[1:]:
            common &= set(self.group_chain(gid))
            if not common:
                return None
        return max(common, key=lambda gid: int(self.group_by_id[gid].get("depth") or 0))

    def node_group_or_root(self, node_id: str) -> str | None:
        group_id = self.node_group.get(node_id)
        if group_id is not None:
            return group_id
        return self.root_group_id


# --------------------------------------------------------------------------- #
# graph walk helpers
# --------------------------------------------------------------------------- #
def _collect_back(ctx: _Ctx, start: str, depth: int = 4) -> list[str]:
    """Breadth-first ancestors of ``start`` (inclusive), up to ``depth`` hops."""
    seen: set[str] = set()
    result: list[str] = []
    queue: list[tuple[str, int]] = [(start, 0)]
    while queue:
        node_id, level = queue.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(node_id)
        if level >= depth:
            continue
        for source in ctx.preds.get(node_id, []):
            if source not in seen:
                queue.append((source, level + 1))
    return result


def _collect_forward(ctx: _Ctx, start: str, depth: int = 4) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    queue: list[tuple[str, int]] = [(start, 0)]
    while queue:
        node_id, level = queue.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(node_id)
        if level >= depth:
            continue
        for target in ctx.succs.get(node_id, []):
            if target not in seen:
                queue.append((target, level + 1))
    return result


def _nearest_weighted(ctx: _Ctx, start: str, kinds: Iterable[str], depth: int = 4) -> str | None:
    wanted = set(kinds)
    if _node_kind(ctx.nodes.get(start, {})) in wanted:
        return start
    for node_id in _collect_back(ctx, start, depth):
        if node_id == start:
            continue
        if _node_kind(ctx.nodes[node_id]) in wanted:
            return node_id
    return None


def _find_back_inclusive(
    ctx: _Ctx,
    start: str,
    predicate: Callable[[str], bool],
    depth: int = 4,
) -> list[str]:
    """Like :func:`_find_back` but also accepts ``start`` itself as match."""
    if start in ctx.nodes and predicate(start):
        return [start]
    return _find_back(ctx, start, predicate, depth)


def _find_back(
    ctx: _Ctx,
    start: str,
    predicate: Callable[[str], bool],
    depth: int = 4,
) -> list[str]:
    """Return the shortest path ``[start, ..., match]`` or ``[]``."""
    queue: list[tuple[str, list[str]]] = [(start, [start])]
    seen: set[str] = {start}
    while queue:
        node_id, path = queue.pop(0)
        if len(path) - 1 > depth:
            continue
        if node_id != start and predicate(node_id):
            return path
        for source in ctx.preds.get(node_id, []):
            if source not in seen:
                seen.add(source)
                queue.append((source, path + [source]))
    return []


def _groups_of_nodes(ctx: _Ctx, node_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        group_id = ctx.node_group.get(node_id)
        if group_id and group_id not in seen:
            seen.add(group_id)
            result.append(group_id)
    return result


def _make_pattern(
    ctx: _Ctx,
    kind: str,
    node_ids: Iterable[str],
    group_ids: Iterable[str] | None = None,
    confidence: float = 0.8,
    evidence: Iterable[str] | None = None,
    extra_tags: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    spec = PATTERN_SPECS.get(kind, {"label": kind, "formula": "?", "tags": [kind]})
    seen: set[str] = set()
    clean_nodes: list[str] = []
    for node_id in node_ids or []:
        if node_id in ctx.nodes and node_id not in seen:
            seen.add(node_id)
            clean_nodes.append(node_id)
    if not clean_nodes:
        return None
    clean_nodes.sort(key=lambda nid: ctx.pos.get(nid, 10**9))
    seen_groups: set[str] = set()
    clean_groups: list[str] = []
    for group_id in group_ids or []:
        if group_id in ctx.group_by_id and group_id not in seen_groups:
            seen_groups.add(group_id)
            clean_groups.append(group_id)
    tags: list[str] = []
    for tag in list(spec.get("tags") or []) + list(extra_tags or []):
        if isinstance(tag, str) and tag and tag not in tags:
            tags.append(tag)
    evidence_list = [str(item) for item in (evidence or []) if item]
    return {
        "kind": kind,
        "label": str(spec.get("label") or kind),
        "formula": str(spec.get("formula") or "?"),
        "nodeIds": clean_nodes,
        "groupIds": clean_groups,
        "confidence": round(min(0.99, max(0.01, float(confidence))), 2),
        "evidence": evidence_list,
        "tags": tags,
    }


# --------------------------------------------------------------------------- #
# pattern detection
# --------------------------------------------------------------------------- #
def _detect_transformer_blocks(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for group in ctx.groups:
        group_id = group["id"]
        subtree = ctx.subtree_nodes(group_id)
        if not subtree:
            continue
        attention = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "attention"]
        if not attention:
            continue
        adds = [
            nid
            for nid in subtree
            if _node_kind(ctx.nodes[nid]) == "add" and len(ctx.preds.get(nid, [])) >= 2
        ]
        norms = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "norm"]
        if not adds:
            continue
        direct = set(ctx.group_direct.get(group_id, []))
        if group.get("kind") != "transformer_block":
            if group.get("kind") == "attention_block":
                continue
            if not (direct & set(attention)):
                continue
            if len(adds) < 2 or not norms:
                continue
        evidence = [
            f"组内 {len(attention)} 个注意力节点",
            f"{len(adds)} 个残差 Add",
            f"{len(norms)} 个归一化层",
        ]
        pattern = _make_pattern(ctx, "transformer_block", subtree, [group_id], 0.95, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_norm_positions(ctx: _Ctx, transformer_patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for block in transformer_patterns:
        if not block.get("groupIds"):
            continue
        group_id = block["groupIds"][0]
        subtree = ctx.subtree_nodes(group_id)
        attention = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "attention"]
        norms = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "norm"]
        adds = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "add"]
        if not attention or not norms:
            continue
        first_attention = min(ctx.pos.get(nid, 10**9) for nid in attention)
        last_add = max((ctx.pos.get(nid, -1) for nid in adds), default=-1)
        pre = [nid for nid in norms if ctx.pos.get(nid, 10**9) < first_attention]
        post = [nid for nid in norms if ctx.pos.get(nid, -1) > last_add]
        if pre:
            pattern = _make_pattern(
                ctx,
                "pre_norm",
                pre + attention[:2],
                [group_id],
                0.9,
                ["归一化层位于注意力子层之前（Pre-LN）", f"LayerNorm {pre[0]} → Attention {attention[0]}"],
            )
            if pattern:
                patterns.append(pattern)
        elif post:
            pattern = _make_pattern(
                ctx,
                "post_norm",
                post + adds[:2],
                [group_id],
                0.9,
                ["归一化层位于残差相加之后（Post-LN）", f"Add {adds[-1]} → LayerNorm {post[-1]}"],
            )
            if pattern:
                patterns.append(pattern)
    return patterns


def _detect_attention_kinds(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {}
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if _node_kind(node) != "attention":
            continue
        group_id = ctx.node_group_or_root(node_id)
        key = group_id or "__root__"
        grouped.setdefault(key, []).append(node_id)

    for key, attention_nodes in grouped.items():
        group_id = None if key == "__root__" else key
        cross = False
        reasons: list[str] = []
        for node_id in attention_nodes:
            sources = ctx.preds.get(node_id, [])
            source_groups: list[str] = []
            for src in sources:
                source_group = ctx.node_group_or_root(src)
                if source_group and source_group not in source_groups:
                    source_groups.append(source_group)
            # Cross attention: Q and K/V come from mutually unrelated
            # branches.  A single source group (even from a previous block)
            # is ordinary sequential self-attention.
            for first_index in range(len(source_groups)):
                for second_index in range(first_index + 1, len(source_groups)):
                    if not ctx.groups_related(source_groups[first_index], source_groups[second_index]):
                        cross = True
                        reasons.append(
                            f"注意力 {node_id} 的 Q/K/V 来自不同分支组 "
                            f"{source_groups[first_index]} / {source_groups[second_index]}"
                        )
                        break
                if cross:
                    break
            if cross:
                break

        scope = list(attention_nodes)
        for node_id in attention_nodes[:2]:
            scope.extend(ctx.preds.get(node_id, [])[:3])
        heads = None
        for node_id in attention_nodes:
            heads = _as_int(_params(ctx.nodes[node_id]).get("numHeads"))
            if heads:
                break
        if cross:
            evidence = reasons[:2] or ["Q 与 K/V 来源不同分支"]
            pattern = _make_pattern(ctx, "cross_attention", scope, [group_id] if group_id else [], 0.85, evidence)
        else:
            evidence = ["Q/K/V 同源（共享同一输入投影）"]
            if heads:
                evidence.append(f"numHeads={heads}")
            pattern = _make_pattern(ctx, "self_attention", scope, [group_id] if group_id else [], 0.9, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _find_gated_mlp(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if not _is_mul(node):
            continue
        sources = ctx.preds.get(node_id, [])
        if len(sources) < 2:
            continue
        branch_linear: list[str] = []
        branch_sigmoid: list[str] = []
        consumed: set[str] = set()
        valid = True
        for source in sources[:2]:
            path = _find_back_inclusive(
                ctx,
                source,
                lambda nid: _node_kind(ctx.nodes[nid]) == "linear",
                depth=4,
            )
            if not path:
                valid = False
                break
            linear = path[-1]
            sigmoid_path = _find_back_inclusive(
                ctx,
                source,
                lambda nid: _is_sigmoid(ctx.nodes[nid]),
                depth=max(1, len(path)),
            )
            if any(_node_kind(ctx.nodes[nid]) == "conv" for nid in path):
                valid = False
                break
            branch_linear.append(linear)
            consumed.update(path)
            if sigmoid_path:
                branch_sigmoid.append(sigmoid_path[-1])
                consumed.update(sigmoid_path)
        if not valid or len(set(branch_linear)) < 2 or not branch_sigmoid:
            continue
        scope = list(consumed) + [node_id]
        group_ids = _groups_of_nodes(ctx, scope)
        evidence = [
            "两条线性分支在 Mul 处逐元素相乘（门控）",
            f"Linear {branch_linear[0]} ⊙ σ({branch_sigmoid[0]})",
        ]
        pattern = _make_pattern(ctx, "gated_mlp", scope, group_ids, 0.85, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_ffn(ctx: _Ctx, gated_patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    gated_nodes: set[str] = set()
    for pattern in gated_patterns:
        gated_nodes.update(pattern.get("nodeIds") or [])
    seen_starts: set[str] = set()
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if _node_kind(node) != "linear" or node_id in seen_starts:
            continue
        params = _params(node)
        out_features = _int_param(params, "outFeatures", "out_features")
        in_features = _int_param(params, "inFeatures", "in_features")
        if not out_features or not in_features or out_features <= in_features * 1.2:
            continue
        # forward BFS to a contracting linear, requiring an activation on the path
        target: str | None = None
        path: list[str] = []
        queue: list[tuple[str, list[str]]] = [(node_id, [node_id])]
        seen: set[str] = {node_id}
        while queue:
            current, current_path = queue.pop(0)
            if len(current_path) - 1 > 6:
                continue
            for target_id in ctx.succs.get(current, []):
                if target_id in seen:
                    continue
                seen.add(target_id)
                next_path = current_path + [target_id]
                target_node = ctx.nodes[target_id]
                kind = _node_kind(target_node)
                if kind == "linear":
                    target_params = _params(target_node)
                    target_in = _int_param(target_params, "inFeatures", "in_features")
                    target_out = _int_param(target_params, "outFeatures", "out_features")
                    has_activation = any(
                        _node_kind(ctx.nodes[path_id]) == "activation" for path_id in next_path
                    )
                    if (
                        target_in == out_features
                        and target_out
                        and target_out <= out_features
                        and has_activation
                    ):
                        target = target_id
                        path = next_path
                        break
                if kind in ("activation", "dropout", "norm", "reshape", "custom", "softmax", "flatten"):
                    queue.append((target_id, next_path))
            if target:
                break
        if target is None:
            continue
        if set(path) & gated_nodes:
            continue
        seen_starts.add(node_id)
        group_ids = _groups_of_nodes(ctx, path)
        if group_ids:
            group_id = ctx.lca_group(group_ids)
            group_ids = [group_id] if group_id else group_ids
        ratio = out_features / in_features if in_features else 0
        evidence = [
            f"Linear {in_features}→{out_features} → 激活 → Linear {out_features}→{_int_param(_params(ctx.nodes[target]), 'outFeatures', 'out_features')}",
            f"隐藏维扩张 {ratio:.1f}×",
        ]
        pattern = _make_pattern(ctx, "ffn", path, group_ids, 0.9, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_residual(
    ctx: _Ctx,
    transformer_patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    transformer_groups: set[str] = set()
    for pattern in transformer_patterns:
        transformer_groups.update(pattern.get("groupIds") or [])
    weight_kinds = {"conv", "deconv", "linear", "ffn", "mlp", "attention"}
    for group in ctx.groups:
        group_id = group["id"]
        if group_id in transformer_groups or group.get("kind") in (
            "transformer_block",
            "attention_block",
            "ffn_block",
        ):
            continue
        subtree = ctx.subtree_nodes(group_id)
        if not subtree:
            continue
        attention_nodes = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "attention"]
        adds = [
            nid
            for nid in ctx.group_direct.get(group_id, [])
            if _node_kind(ctx.nodes[nid]) == "add" and len(ctx.preds.get(nid, [])) >= 2
        ]
        if not adds:
            continue
        if attention_nodes and len(adds) >= 2:
            continue
        has_weight = any(_node_kind(ctx.nodes[nid]) in weight_kinds for nid in subtree)
        if not has_weight:
            continue
        qualified: list[tuple[str, bool, int, str]] = []
        weighted_kinds = {"conv", "deconv", "linear", "attention", "ffn", "mlp"}
        for add_id in adds:
            sources = ctx.preds.get(add_id, [])
            if len(sources) < 2:
                continue
            # A token+position embedding add is not a residual block: at
            # least one branch must carry a weighted operator.
            weighted_branches = 0
            for source in sources:
                if _node_kind(ctx.nodes[source]) in weighted_kinds:
                    weighted_branches += 1
                elif _nearest_weighted(ctx, source, weighted_kinds, depth=4):
                    weighted_branches += 1
            if weighted_branches == 0:
                continue
            positions = [ctx.pos.get(source, 0) for source in sources]
            span = max(positions) - min(positions)
            residual_edges = [
                edge
                for edge in ctx.pred_edges.get(add_id, [])
                if str(edge.get("kind") or "") == "residual"
            ]
            has_residual_edge = bool(residual_edges)
            if has_residual_edge or span >= 2:
                qualified.append((add_id, has_residual_edge, span, residual_edges[0].get("label", "") if residual_edges else ""))
        if not qualified:
            continue
        best = max(qualified, key=lambda item: (item[1], item[2]))
        residual_edge_count = sum(1 for item in qualified if item[1])
        confidence = 0.95 if residual_edge_count else (0.85 if best[2] >= 3 else 0.75)
        evidence = [
            f"Add {best[0]} 汇聚跨层分支（跨度 {best[2]} 层）",
            f"{len(qualified)} 个残差合并点",
        ]
        if residual_edge_count:
            evidence.append(f"{residual_edge_count} 条 kind=residual 跳连边")
        if str(group.get("kind") or "") == "residual_block":
            evidence.append("group.kind=residual_block")
        node_ids = [
            nid
            for nid in subtree
            if _node_kind(ctx.nodes[nid]) in ("conv", "deconv", "linear", "norm", "activation", "add", "dropout", "attention")
        ]
        pattern = _make_pattern(ctx, "residual_block", node_ids or subtree, [group_id], confidence, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_bottleneck(ctx: _Ctx, residual_patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for residual in residual_patterns:
        if not residual.get("groupIds"):
            continue
        group_id = residual["groupIds"][0]
        subtree = ctx.subtree_nodes(group_id)
        convs = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "conv"]
        convs.sort(key=lambda nid: ctx.pos.get(nid, 0))
        found: list[str] | None = None
        for index in range(len(convs) - 2):
            kernels = []
            for offset in range(3):
                kernels.append(_kernel_list(_params(ctx.nodes[convs[index + offset]])))
            if (
                kernels[0]
                and kernels[1]
                and kernels[2]
                and all(dim == 1 for dim in kernels[0])
                and any(dim == 3 for dim in kernels[1])
                and all(dim == 1 for dim in kernels[2])
            ):
                found = convs[index : index + 3]
                break
        if not found:
            continue
        adds = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "add"]
        node_ids = found + adds[:1]
        evidence = [
            "1×1 → 3×3 → 1×1 沙漏结构 + shortcut",
            "通道先压缩再扩张（bottleneck）",
        ]
        pattern = _make_pattern(ctx, "bottleneck", node_ids, [group_id], 0.9, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_inverted_residual(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for group in ctx.groups:
        group_id = group["id"]
        subtree = ctx.subtree_nodes(group_id)
        if not subtree:
            continue
        convs = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "conv"]
        convs.sort(key=lambda nid: ctx.pos.get(nid, 0))
        if len(convs) < 3:
            continue
        depthwise = [
            nid
            for nid in convs
            if _is_depthwise(ctx.nodes[nid]) and ctx.node_group.get(nid) == group_id
        ]
        if not depthwise:
            continue
        adds = [
            nid
            for nid in subtree
            if _node_kind(ctx.nodes[nid]) == "add" and len(ctx.preds.get(nid, [])) >= 2
        ]
        residual_edges = [
            edge
            for nid in subtree
            for edge in ctx.pred_edges.get(nid, [])
            if str(edge.get("kind") or "") == "residual"
        ]
        for depth_id in depthwise:
            depth_params = _params(ctx.nodes[depth_id])
            depth_out = _int_param(depth_params, "outChannels", "out_channels")
            depth_pos = ctx.pos.get(depth_id, 0)
            expand = [
                nid
                for nid in convs
                if ctx.pos.get(nid, 0) < depth_pos
                and _is_pointwise(ctx.nodes[nid])
                and not _is_depthwise(ctx.nodes[nid])
                and _int_param(_params(ctx.nodes[nid]), "outChannels", "out_channels")
                > 2 * max(1, _int_param(_params(ctx.nodes[nid]), "inChannels", "in_channels"))
            ]
            project = [
                nid
                for nid in convs
                if ctx.pos.get(nid, 0) > depth_pos
                and _is_pointwise(ctx.nodes[nid])
                and not _is_depthwise(ctx.nodes[nid])
            ]
            if not expand or not project:
                continue
            expand_id = expand[-1]
            project_id = project[0]
            project_out = _int_param(_params(ctx.nodes[project_id]), "outChannels", "out_channels")
            if depth_out and project_out and project_out >= depth_out:
                continue
            if not adds and not residual_edges:
                continue
            node_ids = [expand_id, depth_id, project_id] + adds[:1]
            evidence = [
                "1×1 扩张 → Depthwise 3×3 → 1×1 投影",
                f"通道 {_int_param(_params(ctx.nodes[expand_id]), 'inChannels', 'in_channels')}→"
                f"{_int_param(_params(ctx.nodes[expand_id]), 'outChannels', 'out_channels')} 扩张",
            ]
            if adds or residual_edges:
                evidence.append("带残差连接")
            pattern = _make_pattern(ctx, "inverted_residual", node_ids, [group_id], 0.9, evidence)
            if pattern:
                patterns.append(pattern)
            break
    return patterns


def _detect_depthwise_separable(
    ctx: _Ctx,
    inverted_patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    inverted_groups: set[str] = set()
    for pattern in inverted_patterns:
        inverted_groups.update(pattern.get("groupIds") or [])
    for group in ctx.groups:
        group_id = group["id"]
        if group_id in inverted_groups:
            continue
        direct = set(ctx.group_direct.get(group_id, []))
        depthwise = [
            nid
            for nid in ctx.subtree_nodes(group_id)
            if nid in direct and _is_depthwise(ctx.nodes[nid])
        ]
        if not depthwise:
            continue
        subtree = ctx.subtree_nodes(group_id)
        for depth_id in depthwise:
            depth_pos = ctx.pos.get(depth_id, 0)
            following = [
                nid
                for nid in subtree
                if ctx.pos.get(nid, 0) > depth_pos
                and _node_kind(ctx.nodes[nid]) == "conv"
                and _is_pointwise(ctx.nodes[nid])
                and not _is_depthwise(ctx.nodes[nid])
            ]
            if not following:
                continue
            point_id = following[0]
            node_ids = [depth_id, point_id]
            evidence = [
                f"Depthwise Conv {depth_id}（groups={_int_param(_params(ctx.nodes[depth_id]), 'groups', default=1)}）"
                f" → 1×1 逐点卷积 {point_id}",
            ]
            pattern = _make_pattern(ctx, "depthwise_separable", node_ids, [group_id], 0.9, evidence)
            if pattern:
                patterns.append(pattern)
            break
    return patterns


def _detect_conv_bn_act(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for group in ctx.groups:
        group_id = group["id"]
        subtree = set(ctx.subtree_nodes(group_id))
        chains: list[list[str]] = []
        for node_id in ctx.group_direct.get(group_id, []):
            if _node_kind(ctx.nodes[node_id]) != "conv":
                continue
            norms = [
                target
                for target in ctx.succs.get(node_id, [])
                if target in subtree and _node_kind(ctx.nodes[target]) == "norm"
            ]
            if not norms:
                continue
            norm_id = norms[0]
            activations = [
                target
                for target in ctx.succs.get(norm_id, [])
                if target in subtree and _node_kind(ctx.nodes[target]) == "activation"
            ]
            if not activations:
                continue
            chains.append([node_id, norm_id, activations[0]])
        if not chains:
            continue
        node_ids: set[str] = set()
        for chain in chains:
            node_ids.update(chain)
        evidence = [
            f"{len(chains)} 条 Conv→Norm→Act 链",
            f"例如 {chains[0][0]}→{chains[0][1]}→{chains[0][2]}",
        ]
        confidence = 0.95 if len(chains) >= 2 else 0.9
        pattern = _make_pattern(ctx, "conv_bn_act", node_ids, [group_id], confidence, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_cbam(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    global_pools = [nid for nid in ctx.order if _is_global_pool(ctx, ctx.nodes[nid])]
    if len(global_pools) < 2:
        return patterns
    sigmoids = [nid for nid in ctx.order if _is_sigmoid(ctx.nodes[nid])]
    if not sigmoids:
        return patterns
    mul_nodes = [
        nid
        for nid in ctx.order
        if _is_mul(ctx.nodes[nid]) and len(ctx.preds.get(nid, [])) >= 2
    ]
    if not mul_nodes:
        return patterns
    spatial_conv = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "conv"
        and any(dim >= 7 for dim in _kernel_list(_params(ctx.nodes[nid])))
    ]
    scope = global_pools[:2] + sigmoids[:1] + mul_nodes[:1] + spatial_conv[:1]
    evidence = [
        f"{len(global_pools)} 个全局池化（avg/max）",
        f"{len(sigmoids)} 个 Sigmoid 门控 + Mul",
    ]
    if spatial_conv:
        evidence.append("含 7×7 空间注意力卷积")
    pattern = _make_pattern(ctx, "cbam", scope, _groups_of_nodes(ctx, scope), 0.8, evidence)
    if pattern:
        patterns.append(pattern)
    return patterns


def _detect_squeeze_excitation(ctx: _Ctx, blocked: set[str]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    global_pools = [nid for nid in ctx.order if _is_global_pool(ctx, ctx.nodes[nid])]
    if not global_pools:
        return patterns
    for mul_id in ctx.order:
        node = ctx.nodes[mul_id]
        if not _is_mul(node) or mul_id in blocked:
            continue
        sources = ctx.preds.get(mul_id, [])
        if len(sources) < 2:
            continue
        sigmoid_path: list[str] = []
        for source in sources:
            path = _find_back_inclusive(ctx, source, lambda nid: _is_sigmoid(ctx.nodes[nid]), depth=4)
            if path:
                sigmoid_path = path
                break
        if not sigmoid_path:
            continue
        sigmoid_id = sigmoid_path[-1]
        back = _collect_back(ctx, sigmoid_id, depth=5)
        pools = [nid for nid in back if _is_global_pool(ctx, ctx.nodes[nid])]
        weighted = [nid for nid in back if _node_kind(ctx.nodes[nid]) in ("linear", "conv")]
        if not pools or len(weighted) < 2:
            continue
        scope = list(dict.fromkeys(back + sigmoid_path + [mul_id]))
        if blocked and set(scope) & blocked:
            continue
        evidence = [
            "全局池化 → 加权降维 → 激活 → 加权升维 → Sigmoid → Mul",
            f"GAP {pools[0]}，门控乘 {mul_id}",
        ]
        pattern = _make_pattern(ctx, "squeeze_excitation", scope, _groups_of_nodes(ctx, scope), 0.9, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_sk(ctx: _Ctx, blocked: set[str]) -> list[dict[str, Any]]:
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    if not ({"sk", "sknet", "selectivekernel", "selective"} & tokens):
        return []
    global_pools = [nid for nid in ctx.order if _is_global_pool(ctx, ctx.nodes[nid])]
    muls = [nid for nid in ctx.order if _is_mul(ctx.nodes[nid])]
    if not global_pools or not muls:
        return []
    kernels: set[tuple[int, ...]] = set()
    for node_id in ctx.order:
        if _node_kind(ctx.nodes[node_id]) == "conv":
            kernel = tuple(_kernel_list(_params(ctx.nodes[node_id])))
            if kernel:
                kernels.add(kernel)
    if len(kernels) < 2:
        return []
    scope = global_pools[:1] + muls[:1]
    pattern = _make_pattern(
        ctx,
        "sk_attention",
        scope,
        _groups_of_nodes(ctx, scope),
        0.72,
        ["SK 命名 + 多卷积核分支 + 全局池化 + 门控相乘", f"卷积核 {sorted(kernels)[:3]}"],
    )
    return [pattern] if pattern else []


def _detect_dense(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for group in ctx.groups:
        group_id = group["id"]
        subtree = set(ctx.subtree_nodes(group_id))
        concats = [
            nid
            for nid in ctx.group_direct.get(group_id, [])
            if _node_kind(ctx.nodes[nid]) == "concat" and len(ctx.preds.get(nid, [])) >= 2
        ]
        if len(concats) < 1:
            continue
        skip_into_concat = any(
            str(edge.get("kind") or "") == "skip"
            for concat_id in concats
            for edge in ctx.pred_edges.get(concat_id, [])
        )
        if skip_into_concat:
            continue
        linked: list[str] = []
        for concat_id in concats:
            for source in ctx.preds.get(concat_id, []):
                if source in concats:
                    linked.append(concat_id)
                    break
                ancestors = _collect_back(ctx, source, depth=2)
                if any(candidate in concats and candidate != concat_id for candidate in ancestors):
                    linked.append(concat_id)
                    break
        if len(concats) < 2 or not linked:
            continue
        convs = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "conv"]
        node_ids = concats + convs
        evidence = [
            f"{len(concats)} 个 concat 串联复用前序特征",
            "DenseNet 式逐层拼接（x_l = H_l([x_0..x_{l-1}]))",
        ]
        pattern = _make_pattern(ctx, "dense_connection", node_ids, [group_id], 0.85, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_unet_skip(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    upsample_nodes = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("deconv", "upsample")]
    if not upsample_nodes:
        return patterns
    plus_concats: list[str] = []
    for concat_id in ctx.order:
        node = ctx.nodes[concat_id]
        if _node_kind(node) != "concat":
            continue
        sources = ctx.preds.get(concat_id, [])
        if len(sources) < 2:
            continue
        skip_edges = [
            edge
            for edge in ctx.pred_edges.get(concat_id, [])
            if str(edge.get("kind") or "") == "skip"
        ]
        upsample_sources = [
            source
            for source in sources
            if _node_kind(ctx.nodes[source]) in ("deconv", "upsample")
            or any(
                _node_kind(ctx.nodes[ancestor]) in ("deconv", "upsample")
                for ancestor in _collect_back(ctx, source, depth=2)
            )
        ]
        if not skip_edges and not upsample_sources:
            continue
        scope = [concat_id] + sources + upsample_sources
        evidence = []
        if skip_edges:
            evidence.append(
                f"{len(skip_edges)} 条 kind=skip 跳连边汇入 concat {concat_id}"
            )
        if upsample_sources:
            evidence.append(f"上采样分支 {upsample_sources[0]} 与编码器特征拼接")
        confidence = 0.95 if skip_edges else 0.8
        pattern = _make_pattern(ctx, "unet_skip", scope, _groups_of_nodes(ctx, scope), confidence, evidence)
        if pattern:
            patterns.append(pattern)
        if len(sources) >= 3 and len(skip_edges) >= 2:
            plus_concats.append(concat_id)
    if plus_concats:
        all_concat: set[str] = set()
        for pattern in patterns:
            if pattern["kind"] == "unet_skip":
                all_concat.update(pattern.get("nodeIds") or [])
        node_ids = sorted(
            [nid for nid in all_concat if _node_kind(ctx.nodes[nid]) == "concat"],
            key=lambda nid: ctx.pos.get(nid, 0),
        )
        plus = _make_pattern(
            ctx,
            "unet_plus_plus",
            node_ids,
            _groups_of_nodes(ctx, node_ids),
            0.8,
            [
                f"{len(node_ids)} 个多输入 skip concat（≥3 路）",
                "嵌套密集跳连（UNet++ 风格）",
            ],
        )
        if plus:
            patterns.append(plus)
    return patterns


def _detect_inception(ctx: _Ctx, skip_concat_ids: set[str]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for concat_id in ctx.order:
        if concat_id in skip_concat_ids:
            continue
        node = ctx.nodes[concat_id]
        if _node_kind(node) != "concat":
            continue
        sources = ctx.preds.get(concat_id, [])
        if len(sources) < 2:
            continue
        if any(str(edge.get("kind") or "") == "skip" for edge in ctx.pred_edges.get(concat_id, [])):
            continue
        if any(_node_kind(ctx.nodes[source]) in ("deconv", "upsample") for source in sources):
            continue
        # common ancestor of all branches
        ancestor_sets = [set(_collect_back(ctx, source, depth=12)) for source in sources]
        common = set.intersection(*ancestor_sets) if ancestor_sets else set()
        common.discard(concat_id)
        common = {
            nid
            for nid in common
            if _node_kind(ctx.nodes[nid]) != "output"
        }
        if not common:
            continue
        anchor = max(common, key=lambda nid: ctx.pos.get(nid, 0))
        branch_kernels: list[list[tuple[int, ...]]] = []
        branch_convs: list[str] = []
        valid = True
        for source in sources:
            path = _find_back(
                ctx,
                source,
                lambda nid: nid == anchor,
                depth=12,
            )
            if not path:
                valid = False
                break
            kernels: list[tuple[int, ...]] = []
            for node_id in path:
                if _node_kind(ctx.nodes[node_id]) == "conv":
                    kernel = tuple(_kernel_list(_params(ctx.nodes[node_id])))
                    if kernel:
                        kernels.append(kernel)
                        branch_convs.append(node_id)
            if not kernels:
                valid = False
                break
            branch_kernels.append(kernels)
        if not valid:
            continue
        distinct = {kernel for kernels in branch_kernels for kernel in kernels}
        if len(distinct) < 2 or len(sources) < 2:
            continue
        scope = [anchor, concat_id] + sources + branch_convs
        evidence = [
            f"同一输入 {anchor} 分出 {len(sources)} 条卷积分支",
            "不同 kernel " + "/".join(sorted({_kernel_to_text(kernel) for kernel in distinct})),
            f"分支在 concat {concat_id} 汇合",
        ]
        pattern = _make_pattern(
            ctx,
            "inception",
            scope,
            _groups_of_nodes(ctx, scope),
            0.85,
            evidence,
        )
        if pattern:
            patterns.append(pattern)
    return patterns


def _kernel_to_text(kernel: Iterable[int]) -> str:
    return "x".join(str(max(1, dim)) for dim in kernel)


def _detect_fpn(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    upsample_nodes = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("deconv", "upsample")]
    if not upsample_nodes:
        return patterns
    lateral_adds: list[str] = []
    for node_id in ctx.order:
        if _node_kind(ctx.nodes[node_id]) != "add":
            continue
        sources = ctx.preds.get(node_id, [])
        if len(sources) < 2:
            continue
        up_branches = [
            source
            for source in sources
            if _node_kind(ctx.nodes[source]) in ("deconv", "upsample")
            or any(
                _node_kind(ctx.nodes[ancestor]) in ("deconv", "upsample")
                for ancestor in _collect_back(ctx, source, depth=2)
            )
        ]
        lateral_branches = [
            source
            for source in sources
            if _node_kind(ctx.nodes[source]) == "conv"
            or _nearest_weighted(ctx, source, {"conv"}, depth=2)
        ]
        if up_branches and lateral_branches:
            lateral_adds.append(node_id)
    if not lateral_adds:
        return patterns
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    panet = bool({"panet", "bifpn"} & tokens)
    kind = "panet" if panet else "fpn"
    scope: set[str] = set()
    for add_id in lateral_adds[:6]:
        scope.add(add_id)
        scope.update(ctx.preds.get(add_id, []))
    if panet:
        evidence = [
            f"{len(lateral_adds)} 处自顶向下 + 自底向上双向融合",
            "PANet/BiFPN 命名与横向连接结构",
        ]
    else:
        evidence = [
            f"{len(lateral_adds)} 处上采样分支与横向卷积分支在 Add 融合",
            "多尺度特征金字塔（FPN）",
        ]
    pattern = _make_pattern(ctx, kind, scope, _groups_of_nodes(ctx, scope), 0.8, evidence)
    if pattern:
        patterns.append(pattern)
    return patterns


def _detect_rnn(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    rnn_nodes = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "rnn"]
    if not rnn_nodes:
        return patterns
    grouped: dict[str, list[str]] = {}
    for node_id in rnn_nodes:
        group_id = ctx.node_group_or_root(node_id) or "__root__"
        grouped.setdefault(group_id, []).append(node_id)
    for key, members in grouped.items():
        group_id = None if key == "__root__" else key
        op = _op_short(ctx.nodes[members[0]])
        if "lstm" in op.lower():
            formula = "(h_t, c_t) = LSTM(x_t, (h_{t-1}, c_{t-1}))"
        elif "gru" in op.lower():
            formula = "h_t = GRU(x_t, h_{t-1})"
        else:
            formula = "h_t = RNN(x_t, h_{t-1})"
        spec = dict(PATTERN_SPECS["rnn_block"])
        local = dict(spec)
        local["formula"] = formula
        # temporarily override the catalogue formula via a shallow copy
        saved = PATTERN_SPECS["rnn_block"]["formula"]
        PATTERN_SPECS["rnn_block"]["formula"] = formula
        try:
            pattern = _make_pattern(
                ctx,
                "rnn_block",
                members,
                [group_id] if group_id else [],
                0.95,
                [f"{op} 循环层 ×{len(members)}", "按时间步维护隐藏状态"],
            )
        finally:
            PATTERN_SPECS["rnn_block"]["formula"] = saved
        if pattern:
            patterns.append(pattern)
    return patterns


def _detect_moe(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    name_hint = bool({"moe", "expert", "experts", "router", "gating", "mixtral", "switch"} & tokens)
    for merge_id in ctx.order:
        merge_node = ctx.nodes[merge_id]
        kind = _node_kind(merge_node)
        if kind not in ("add", "concat") and not _is_mul(merge_node):
            continue
        sources = ctx.preds.get(merge_id, [])
        if len(sources) < 2:
            continue
        experts: list[str] = []
        for source in sources:
            linear = _nearest_weighted(ctx, source, {"linear", "ffn", "mlp"}, depth=3)
            if linear:
                experts.append(linear)
        experts = list(dict.fromkeys(experts))
        if len(experts) < 2:
            continue
        # A gate is a softmax router in the merge ancestry that is *not* part
        # of any expert path (otherwise it is just an attention softmax).
        gate = False
        for ancestor in _collect_back(ctx, merge_id, depth=4):
            if _node_kind(ctx.nodes[ancestor]) != "softmax":
                continue
            if all(ancestor in _collect_back(ctx, expert, depth=6) for expert in experts):
                continue
            gate = True
            break
        if not gate and not name_hint:
            continue
        if not (kind in ("add", "concat") or gate or name_hint):
            continue
        scope = experts + sources + [merge_id]
        evidence = [
            f"{len(experts)} 个并行专家分支",
            f"合并点 {merge_id}（{kind}）",
        ]
        if gate:
            evidence.append("Softmax 路由门控")
        if name_hint:
            evidence.append("名称包含 expert/router/moe")
        pattern = _make_pattern(ctx, "moe", scope, _groups_of_nodes(ctx, scope), 0.85 if gate else 0.8, evidence)
        if pattern:
            patterns.append(pattern)
    return patterns


def _find_gated(ctx: _Ctx, merge_id: str) -> str | None:
    """Return a sigmoid branch id when ``merge_id`` combines two linear branches."""
    sources = ctx.preds.get(merge_id, [])
    if len(sources) < 2:
        return None
    linears: list[str] = []
    sigmoid: str | None = None
    for source in sources[:2]:
        path = _find_back_inclusive(ctx, source, lambda nid: _node_kind(ctx.nodes[nid]) == "linear", depth=4)
        if not path:
            return None
        linears.append(path[-1])
        sig = _find_back_inclusive(ctx, source, lambda nid: _is_sigmoid(ctx.nodes[nid]), depth=max(1, len(path)))
        if sig:
            sigmoid = sig[-1]
    if len(set(linears)) < 2:
        return None
    return sigmoid


def _detect_ssm(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"mamba", "ssm", "selectivescan", "statespace", "s4", "mamba2"} & tokens)
    sequence_conv = any(
        _node_kind(ctx.nodes[nid]) == "conv" and _rank(ctx.nodes[nid]) == 3
        for nid in ctx.order
    )
    gate_mul = any(_is_mul(ctx.nodes[nid]) and len(ctx.preds.get(nid, [])) >= 2 for nid in ctx.order)
    linears = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "linear"]
    if not hint and not (sequence_conv and gate_mul and len(linears) >= 2):
        return patterns
    scope = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("conv", "linear", "activation") or _is_mul(ctx.nodes[nid])][:12]
    evidence = ["SSM/Mamba 命名命中" if hint else "序列卷积 + 门控乘 + 线性投影结构"]
    if sequence_conv:
        evidence.append("1D 序列卷积")
    if gate_mul:
        evidence.append("门控逐元素相乘")
    pattern = _make_pattern(ctx, "ssm", scope, _groups_of_nodes(ctx, scope), 0.9 if hint else 0.7, evidence)
    return [pattern] if pattern else []


def _detect_rwkv(ctx: _Ctx) -> list[dict[str, Any]]:
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"rwkv", "timemix", "receptance", "timefirst"} & tokens)
    if not hint:
        return []
    sigmoids = [nid for nid in ctx.order if _is_sigmoid(ctx.nodes[nid])]
    muls = [nid for nid in ctx.order if _is_mul(ctx.nodes[nid]) and len(ctx.preds.get(nid, [])) >= 2]
    if not sigmoids or not muls:
        return []
    scope = sigmoids[:1] + muls[:1]
    pattern = _make_pattern(
        ctx,
        "rwkv",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85,
        ["RWKV/time-mix 命名命中", "Sigmoid 门控 + 逐元素乘（time-mix）"],
    )
    return [pattern] if pattern else []


def _detect_gnn(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"gnn", "gcn", "gat", "graphsage", "graph", "message", "edge"} & tokens)
    no_conv = ctx.kind_counts.get("conv", 0) == 0
    no_attention = ctx.kind_counts.get("attention", 0) == 0
    linear_count = ctx.kind_counts.get("linear", 0) + ctx.kind_counts.get("mlp", 0)
    has_merge = ctx.kind_counts.get("add", 0) + ctx.kind_counts.get("concat", 0) > 0
    has_norm = ctx.kind_counts.get("norm", 0) > 0
    structural = no_conv and no_attention and linear_count >= 2 and has_merge and has_norm
    if not (hint or structural):
        return patterns
    scope = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("linear", "norm", "add", "concat")][:10]
    if not scope:
        scope = ctx.node_ids[:10]
    evidence = []
    if hint:
        evidence.append("GNN/graph/message 命名命中")
    if structural:
        evidence.append("线性传播 + 邻域聚合（add/concat）+ 归一化")
    pattern = _make_pattern(
        ctx,
        "gnn_message_passing",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85 if hint else 0.65,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_point_mlp(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"pointnet", "pointmlp", "setabstraction", "pointcloud", "point"} & tokens)
    no_conv = ctx.kind_counts.get("conv", 0) == 0
    no_attention = ctx.kind_counts.get("attention", 0) == 0
    linears = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "linear"]
    global_max = [nid for nid in ctx.order if _is_global_max_pool(ctx, ctx.nodes[nid])]
    structural = no_conv and no_attention and len(linears) >= 2 and bool(global_max)
    if not (hint or structural):
        return patterns
    scope = linears[:4] + global_max[:2]
    evidence = []
    if hint:
        evidence.append("PointNet/点云命名命中")
    if global_max:
        evidence.append("逐点 MLP 后接全局最大池化")
    pattern = _make_pattern(
        ctx,
        "point_mlp",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85 if hint else 0.65,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_nerf(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"nerf", "viewdir", "ray", "radiance", "implicitfield"} & tokens)
    linears = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "linear"]
    positional = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "positional"]
    output = _first_shape(ctx.model.get("outputSpec")[0]) if isinstance(ctx.model.get("outputSpec"), list) and ctx.model.get("outputSpec") else None
    output_last = output[-1] if output else None
    structural = (
        ctx.kind_counts.get("conv", 0) == 0
        and len(linears) >= 4
        and bool(positional)
        and output_last == 4
    )
    if not (hint or structural):
        return patterns
    scope = positional[:2] + linears[:6]
    evidence = []
    if hint:
        evidence.append("NeRF/radiance/view-dir 命名命中")
    if structural:
        evidence.append("位置编码 + 多层 MLP → (σ, RGB)")
    pattern = _make_pattern(
        ctx,
        "nerf_mlp",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85 if hint else 0.7,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_vae(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"vae", "variational", "reparameterization", "reparam", "mu", "logvar"} & tokens)
    structural = False
    scope: list[str] = []
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if _node_kind(node) != "reshape":
            continue
        op = _op_short(node).lower()
        if op not in ("chunk", "split", "unflatten") and "chunk" not in str(node.get("name") or "").lower():
            continue
        sources = ctx.preds.get(node_id, [])
        if not sources:
            continue
        if _node_kind(ctx.nodes[sources[0]]) != "linear":
            continue
        merge_candidates = [
            candidate
            for candidate in _collect_forward(ctx, node_id, depth=4)
            if (_node_kind(ctx.nodes[candidate]) in ("add", "concat") or _is_mul(ctx.nodes[candidate]))
            and len(ctx.preds.get(candidate, [])) >= 2
        ]
        if not merge_candidates:
            continue
        structural = True
        scope = _collect_back(ctx, merge_candidates[0], depth=5) + [node_id, merge_candidates[0]]
        break
    if not (hint or structural):
        return patterns
    if not scope:
        scope = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("linear", "reshape", "add")][:8]
    evidence = []
    if hint:
        evidence.append("VAE/reparameterization 命名命中")
    if structural:
        evidence.append("线性输出切分为均值/方差 → 乘/加重参数化")
    pattern = _make_pattern(
        ctx,
        "vae_reparam",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85 if hint else 0.72,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_diffusion(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    hint = bool({"diffusion", "ddpm", "ddim", "laten diffusion", "stablediffusion", "denoiser"} & tokens) or "diffusion" in _norm_text(ctx.model.get("name")) or "diffusion" in _norm_text(ctx.model.get("className"))
    unet_skip = ctx.kind_counts.get("concat", 0) > 0 and any(
        str(edge.get("kind") or "") == "skip" for edge in ctx.edges
    )
    upsample = any(_node_kind(ctx.nodes[nid]) in ("deconv", "upsample") for nid in ctx.order)
    linears = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "linear"]
    structural = unet_skip and upsample and bool(linears)
    if not (hint or structural):
        return patterns
    if not (unet_skip or upsample):
        return patterns
    scope = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) in ("conv", "deconv", "upsample", "linear", "concat")][:12]
    evidence = []
    if hint:
        evidence.append("Diffusion/DDPM 命名命中")
    if unet_skip:
        evidence.append("UNet skip 连接")
    if linears:
        evidence.append("时间步嵌入 MLP")
    pattern = _make_pattern(
        ctx,
        "diffusion_unet",
        scope,
        _groups_of_nodes(ctx, scope),
        0.88 if hint else 0.72,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_swin(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    text = ""
    for node in list(ctx.nodes.values()) + [ctx.model]:
        if not isinstance(node, dict):
            continue
        for key in ("name", "modulePath", "op", "className", "label"):
            value = node.get(key)
            if isinstance(value, str):
                text += " " + value
    normalized = _norm_text(text)
    hint = bool({"swin", "window", "shiftedwindow", "windowmsa", "roll"} & set(_name_tokens(ctx.nodes.values(), ctx.model))) or "swin" in normalized
    attention = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "attention"]
    if not attention:
        return patterns
    window_ops = [
        nid
        for nid in ctx.order
        if "roll" in _op_short(ctx.nodes[nid]).lower()
        or "window" in _norm_text(ctx.nodes[nid].get("name"))
    ]
    structural = len(attention) >= 2 and bool(window_ops)
    if not (hint or structural):
        return patterns
    scope = attention[:4] + window_ops[:4]
    evidence = []
    if hint:
        evidence.append("Swin/window 命名命中")
    if structural:
        evidence.append("注意力前含窗口切分/roll 位移")
    pattern = _make_pattern(
        ctx,
        "swin_attention",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85 if hint else 0.7,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_rope(ctx: _Ctx) -> list[dict[str, Any]]:
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    if not ({"rope", "rotary", "rotaryemb"} & tokens):
        return []
    rope_nodes = [
        nid
        for nid in ctx.order
        if "rope" in _norm_text(ctx.nodes[nid].get("modulePath"))
        or "rotary" in _norm_text(ctx.nodes[nid].get("modulePath"))
        or "rope" in _norm_text(ctx.nodes[nid].get("name"))
    ]
    attention = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "attention"]
    scope = rope_nodes or attention[:2]
    if not scope:
        scope = ctx.node_ids[:4]
    pattern = _make_pattern(
        ctx,
        "rope",
        scope,
        _groups_of_nodes(ctx, scope),
        0.85,
        ["RoPE/rotary 命名命中", "对 Q/K 施加旋转位置编码"],
    )
    return [pattern] if pattern else []


def _detect_rmsnorm(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    nodes = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "norm"
        and (
            "rms" in _op_short(ctx.nodes[nid]).lower()
            or "rms" in _norm_text(ctx.nodes[nid].get("name"))
        )
    ]
    if not nodes:
        return patterns
    pattern = _make_pattern(
        ctx,
        "rmsnorm",
        nodes,
        _groups_of_nodes(ctx, nodes),
        0.95,
        [f"RMSNorm ×{len(nodes)}", "仅按均方根缩放，无均值中心化"],
    )
    return [pattern] if pattern else []


def _detect_lora(ctx: _Ctx) -> list[dict[str, Any]]:
    nodes = [
        nid
        for nid in ctx.order
        if "lora" in _norm_text(ctx.nodes[nid].get("modulePath"))
        or "lora" in _norm_text(ctx.nodes[nid].get("name"))
    ]
    if not nodes:
        return []
    pattern = _make_pattern(
        ctx,
        "lora",
        nodes,
        _groups_of_nodes(ctx, nodes),
        0.9,
        [f"LoRA 模块 ×{len(nodes)}", "低秩增量 W + α/r·BA"],
    )
    return [pattern] if pattern else []


def _detect_patch_embedding(ctx: _Ctx) -> list[dict[str, Any]]:
    attention = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "attention"]
    if not attention:
        return []
    first_attention = min(ctx.pos.get(nid, 10**9) for nid in attention)
    patches = []
    for node_id in ctx.order:
        if ctx.pos.get(node_id, 10**9) >= first_attention:
            continue
        node = ctx.nodes[node_id]
        if _node_kind(node) != "conv":
            continue
        params = _params(node)
        kernel = _kernel_list(params)
        stride = _as_int_list(_param(params, "stride", "strides"))
        if kernel and stride and kernel == stride and any(dim > 1 for dim in kernel):
            patches.append(node_id)
    if not patches:
        return []
    pattern = _make_pattern(
        ctx,
        "patch_embedding",
        patches,
        _groups_of_nodes(ctx, patches),
        0.82,
        [f"k=stride 卷积层 ×{len(patches)}（分块嵌入）", "位于 Transformer 注意力之前"],
    )
    return [pattern] if pattern else []


def _detect_classifier_head(ctx: _Ctx) -> list[dict[str, Any]]:
    candidates = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) in ("linear", "mlp")
        and _first_shape(ctx.nodes[nid]) is not None
    ]
    if not candidates:
        return []
    last = candidates[-1]
    out_shape = _first_shape(ctx.nodes[last]) or []
    has_features = any(
        _node_kind(ctx.nodes[nid]) in ("conv", "attention", "rnn", "embedding")
        for nid in ctx.order
    )
    if not has_features and len(candidates) < 2:
        return []
    out_features = _int_param(_params(ctx.nodes[last]), "outFeatures", "out_features")
    evidence = [f"末端 Linear 输出 {out_features or '?'} 维 logits"]
    if len(out_shape) == 3:
        evidence.append("序列输出（LM head）")
    pattern = _make_pattern(
        ctx,
        "classifier_head",
        [last],
        _groups_of_nodes(ctx, [last]),
        0.85,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_global_pool(ctx: _Ctx) -> list[dict[str, Any]]:
    nodes = [nid for nid in ctx.order if _is_global_pool(ctx, ctx.nodes[nid])]
    if not nodes:
        return []
    pattern = _make_pattern(
        ctx,
        "global_pool",
        nodes,
        _groups_of_nodes(ctx, nodes),
        0.9,
        [f"全局池化 ×{len(nodes)}", "空间维压缩为 1×1 后进入分类头"],
    )
    return [pattern] if pattern else []


def _detect_positional(ctx: _Ctx) -> list[dict[str, Any]]:
    nodes: list[str] = []
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if _node_kind(node) == "positional":
            nodes.append(node_id)
            continue
        text = _norm_text(node.get("name")) + " " + _norm_text(node.get("modulePath"))
        if "posemb" in text or "positional" in text or "positionembed" in text or "posencoding" in text:
            nodes.append(node_id)
    # sinusoidal buffer case: Embedding -> Mul(scale) -> Add(position)
    sinusoidal = False
    for node_id in ctx.order:
        if _node_kind(ctx.nodes[node_id]) != "add":
            continue
        for source in ctx.preds.get(node_id, []):
            source_node = ctx.nodes[source]
            source_preds = ctx.preds.get(source, [])
            if (
                _is_mul(source_node)
                and len(source_preds) == 1
                and _node_kind(ctx.nodes[source_preds[0]]) == "embedding"
            ):
                sinusoidal = True
                for extra in (source, node_id):
                    if extra not in nodes:
                        nodes.append(extra)
    if not nodes:
        return []
    evidence = [f"位置编码节点 ×{len(nodes)}", "向 token 表示注入顺序信息"]
    if sinusoidal:
        evidence[0] = "嵌入缩放 + 位置编码相加（PE）"
    pattern = _make_pattern(
        ctx,
        "positional_encoding",
        nodes,
        _groups_of_nodes(ctx, nodes),
        0.85,
        evidence,
    )
    return [pattern] if pattern else []


def _detect_weight_tying(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    # 1) explicit modulePath reuse between weighted nodes
    by_path: dict[str, list[str]] = {}
    weighted_kinds = {"conv", "deconv", "linear", "embedding", "attention", "rnn", "ffn", "mlp"}
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        if _node_kind(node) not in weighted_kinds:
            continue
        if _as_int(node.get("numParams"), 0) == 0:
            continue
        path = str(node.get("modulePath") or "")
        if path:
            by_path.setdefault(path, []).append(node_id)
    for path, nodes in by_path.items():
        if len(nodes) < 2:
            continue
        pattern = _make_pattern(
            ctx,
            "weight_tying",
            nodes,
            _groups_of_nodes(ctx, nodes),
            0.9,
            [f"节点 {', '.join(nodes)} 共享 modulePath {path!r}", "同一权重被复用"],
        )
        if pattern:
            patterns.append(pattern)
    # 2) embedding <-> transposed linear head (LM head / tied embeddings)
    embeddings = [nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) == "embedding"]
    for linear_id in ctx.order:
        node = ctx.nodes[linear_id]
        if _node_kind(node) != "linear":
            continue
        params = _params(node)
        if params.get("bias") is not False:
            continue
        out_features = _int_param(params, "outFeatures", "out_features")
        in_features = _int_param(params, "inFeatures", "in_features")
        if not out_features or not in_features:
            continue
        for embed_id in embeddings:
            embed_params = _params(ctx.nodes[embed_id])
            vocab = _int_param(embed_params, "numEmbeddings", "num_embeddings")
            dim = _int_param(embed_params, "embeddingDim", "embedding_dim")
            if vocab == out_features and dim == in_features:
                pattern = _make_pattern(
                    ctx,
                    "weight_tying",
                    [embed_id, linear_id],
                    _groups_of_nodes(ctx, [embed_id, linear_id]),
                    0.85,
                    [
                        f"Linear({in_features}→{out_features}, bias=False) 与 Embedding({vocab},{dim}) 维度转置一致",
                        "词嵌入矩阵与输出头共享权重",
                    ],
                )
                if pattern:
                    patterns.append(pattern)
    return patterns


def _detect_patterns(ctx: _Ctx) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    transformer = _detect_transformer_blocks(ctx)
    patterns.extend(transformer)
    patterns.extend(_detect_norm_positions(ctx, transformer))
    patterns.extend(_detect_attention_kinds(ctx))
    gated = _find_gated_mlp(ctx)
    patterns.extend(gated)
    patterns.extend(_detect_ffn(ctx, gated))
    residual = _detect_residual(ctx, transformer)
    patterns.extend(residual)
    patterns.extend(_detect_bottleneck(ctx, residual))
    inverted = _detect_inverted_residual(ctx)
    patterns.extend(inverted)
    patterns.extend(_detect_depthwise_separable(ctx, inverted))
    patterns.extend(_detect_conv_bn_act(ctx))
    cbam = _detect_cbam(ctx)
    patterns.extend(cbam)
    blocked = {nid for pattern in cbam for nid in pattern.get("nodeIds") or []}
    patterns.extend(_detect_squeeze_excitation(ctx, blocked))
    patterns.extend(_detect_sk(ctx, blocked))
    unet = _detect_unet_skip(ctx)
    patterns.extend(unet)
    skip_concat_ids = {
        nid
        for pattern in unet
        if pattern.get("kind") == "unet_skip"
        for nid in pattern.get("nodeIds") or []
        if _node_kind(ctx.nodes.get(nid, {})) == "concat"
    }
    patterns.extend(_detect_dense(ctx))
    patterns.extend(_detect_inception(ctx, skip_concat_ids))
    patterns.extend(_detect_fpn(ctx))
    patterns.extend(_detect_rnn(ctx))
    patterns.extend(_detect_moe(ctx))
    patterns.extend(_detect_ssm(ctx))
    patterns.extend(_detect_rwkv(ctx))
    patterns.extend(_detect_gnn(ctx))
    patterns.extend(_detect_point_mlp(ctx))
    patterns.extend(_detect_nerf(ctx))
    patterns.extend(_detect_vae(ctx))
    patterns.extend(_detect_diffusion(ctx))
    patterns.extend(_detect_swin(ctx))
    patterns.extend(_detect_rope(ctx))
    patterns.extend(_detect_rmsnorm(ctx))
    patterns.extend(_detect_lora(ctx))
    patterns.extend(_detect_patch_embedding(ctx))
    patterns.extend(_detect_classifier_head(ctx))
    patterns.extend(_detect_global_pool(ctx))
    patterns.extend(_detect_positional(ctx))
    patterns.extend(_detect_weight_tying(ctx))

    # de-duplicate identical detections and cap
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for pattern in patterns:
        if not pattern:
            continue
        key = (
            pattern["kind"],
            tuple(pattern.get("nodeIds") or []),
            tuple(pattern.get("groupIds") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(pattern)
    return unique[:MAX_PATTERNS]


def _finalize_patterns(ctx: _Ctx, patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def first_position(pattern: dict[str, Any]) -> int:
        positions = [ctx.pos.get(nid, 10**9) for nid in pattern.get("nodeIds") or []]
        return min(positions) if positions else 10**9

    ordered = sorted(
        patterns,
        key=lambda pattern: (
            first_position(pattern),
            pattern.get("kind", ""),
            tuple(pattern.get("nodeIds") or []),
        ),
    )
    finalized: list[dict[str, Any]] = []
    for index, pattern in enumerate(ordered, start=1):
        item = dict(pattern)
        item["id"] = f"p{index}"
        finalized.append(item)
    return finalized


def _group_patterns(ctx: _Ctx, patterns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, tuple[int, float, dict[str, Any]]] = {}
    for pattern in patterns:
        priority = _GROUP_PATTERN_PRIORITY.get(pattern.get("kind", ""), 10)
        confidence = float(pattern.get("confidence") or 0)
        for group_id in pattern.get("groupIds") or []:
            current = best.get(group_id)
            candidate = (priority, confidence, pattern)
            if current is None or candidate[:2] > current[:2]:
                best[group_id] = candidate
    result: dict[str, dict[str, Any]] = {}
    for group_id, (_, _, pattern) in best.items():
        result[group_id] = {
            "kind": pattern["kind"],
            "label": pattern["label"],
            "formula": pattern["formula"],
            "confidence": pattern["confidence"],
            "tags": list(pattern.get("tags") or []),
        }
    return result


# --------------------------------------------------------------------------- #
# structural facts + knowledge base scoring
# --------------------------------------------------------------------------- #
_KIND_LABELS = {
    "conv": "卷积层",
    "deconv": "转置卷积",
    "linear": "线性层",
    "mlp": "MLP",
    "norm": "归一化层",
    "activation": "激活函数",
    "pool": "池化层",
    "dropout": "Dropout",
    "flatten": "展平",
    "reshape": "形状变换",
    "attention": "注意力",
    "ffn": "前馈网络",
    "embedding": "嵌入层",
    "positional": "位置编码",
    "add": "逐元素相加",
    "concat": "拼接",
    "upsample": "上采样",
    "rnn": "循环层",
    "softmax": "Softmax",
    "block": "复合块",
    "container": "容器",
    "custom": "自定义算子",
    "unknown": "未知算子",
}

_COMPONENT_SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "conv": ("conv",),
    "convolution": ("conv",),
    "conv_bn_act": ("conv_bn_act",),
    "conv_bn_relu": ("conv_bn_act", "activation"),
    "residual": ("residual",),
    "residual_block": ("residual",),
    "skip": ("skip",),
    "skip_connection": ("skip", "residual"),
    "bottleneck": ("bottleneck",),
    "depthwise": ("depthwise",),
    "depthwise_conv": ("depthwise",),
    "depthwise_separable": ("depthwise_separable",),
    "inverted_residual": ("inverted_residual",),
    "mobilenet_block": ("inverted_residual",),
    "se": ("se",),
    "squeeze_excitation": ("se",),
    "squeeze_and_excitation": ("se",),
    "channel_attention": ("se", "cbam"),
    "cbam": ("cbam",),
    "sk": ("sk",),
    "sk_attention": ("sk",),
    "selective_kernel": ("sk",),
    "inception": ("inception",),
    "dense": ("dense",),
    "dense_connection": ("dense",),
    "densenet_block": ("dense",),
    "unet_skip": ("unet_skip",),
    "unet_plus_plus": ("unet_plus_plus", "unet_skip", "dense"),
    "dense_skip": ("unet_plus_plus", "unet_skip"),
    "fpn": ("fpn",),
    "panet": ("panet",),
    "bifpn": ("fpn",),
    "transformer_block": ("transformer_block",),
    "transformer_layer": ("transformer_block",),
    "self_attention": ("self_attention",),
    "cross_attention": ("cross_attention",),
    "attention": ("attention",),
    "mha": ("attention",),
    "multi_head_attention": ("attention",),
    "ffn": ("ffn",),
    "feed_forward": ("ffn",),
    "gated_mlp": ("gated_mlp",),
    "swiglu": ("gated_mlp",),
    "geglu": ("gated_mlp",),
    "glu": ("gated_mlp",),
    "pre_norm": ("pre_norm",),
    "post_norm": ("post_norm",),
    "rmsnorm": ("rmsnorm",),
    "layernorm": ("layernorm",),
    "batchnorm": ("batchnorm",),
    "norm": ("norm",),
    "normalization": ("norm",),
    "embedding": ("embedding",),
    "token_embedding": ("embedding",),
    "positional": ("positional",),
    "positional_encoding": ("positional",),
    "rope": ("rope",),
    "rotary": ("rope",),
    "rnn": ("rnn",),
    "lstm": ("rnn",),
    "gru": ("rnn",),
    "recurrent": ("rnn",),
    "moe": ("moe",),
    "mixture_of_experts": ("moe",),
    "expert": ("moe",),
    "router": ("moe",),
    "ssm": ("ssm",),
    "mamba": ("ssm",),
    "state_space": ("ssm",),
    "rwkv": ("rwkv",),
    "time_mix": ("rwkv",),
    "gnn": ("gnn",),
    "message_passing": ("gnn",),
    "graph_conv": ("gnn",),
    "gcn": ("gnn",),
    "gat": ("attention", "gnn"),
    "point_mlp": ("point_mlp",),
    "pointnet": ("point_mlp",),
    "point_encoder": ("point_mlp",),
    "nerf": ("nerf",),
    "nerf_mlp": ("nerf",),
    "view_mlp": ("nerf",),
    "vae": ("vae",),
    "vae_reparam": ("vae",),
    "reparameterization": ("vae",),
    "diffusion": ("diffusion",),
    "diffusion_unet": ("diffusion", "unet_skip"),
    "swin": ("swin",),
    "swin_attention": ("swin",),
    "window_attention": ("swin",),
    "lora": ("lora",),
    "low_rank_adapter": ("lora",),
    "patch_embedding": ("patch_embed",),
    "patch_embed": ("patch_embed",),
    "classifier_head": ("classifier_head",),
    "head": ("classifier_head",),
    "prediction_head": ("classifier_head",),
    "global_pool": ("global_pool",),
    "global_average_pooling": ("global_pool",),
    "gap": ("global_pool",),
    "pool": ("pool",),
    "max_pool": ("pool",),
    "avg_pool": ("pool",),
    "upsample": ("upsample",),
    "deconv": ("deconv",),
    "conv_transpose": ("deconv",),
    "concat": ("concat",),
    "concatenation": ("concat",),
    "add": ("add",),
    "elementwise_add": ("add",),
    "linear": ("linear",),
    "dense_layer": ("linear",),
    "mlp": ("mlp",),
    "perceptron": ("mlp", "linear"),
    "dropout": ("dropout",),
    "softmax": ("softmax",),
    "activation": ("activation",),
    "relu": ("activation",),
    "gelu": ("activation",),
    "silu": ("activation",),
    "swish": ("activation",),
    "weight_tying": ("weight_tying",),
    "tied_weights": ("weight_tying",),
    "tied_embedding": ("weight_tying",),
}

_TOKEN_SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "conv": ("conv",),
    "convolution": ("conv",),
    "transposed": ("deconv",),
    "deconv": ("deconv",),
    "linear": ("linear",),
    "mlp": ("mlp",),
    "norm": ("norm",),
    "normalization": ("norm",),
    "batchnorm": ("batchnorm",),
    "layernorm": ("layernorm",),
    "rmsnorm": ("rmsnorm",),
    "attention": ("attention",),
    "attn": ("attention",),
    "residual": ("residual",),
    "skip": ("skip",),
    "bottleneck": ("bottleneck",),
    "depthwise": ("depthwise",),
    "inverted": ("inverted_residual",),
    "se": ("se",),
    "squeeze": ("se",),
    "excitation": ("se",),
    "cbam": ("cbam",),
    "sk": ("sk",),
    "inception": ("inception",),
    "dense": ("dense",),
    "unet": ("unet_skip",),
    "fpn": ("fpn",),
    "panet": ("panet",),
    "transformer": ("transformer_block",),
    "self": ("self_attention",),
    "cross": ("cross_attention",),
    "ffn": ("ffn",),
    "feedforward": ("ffn",),
    "gated": ("gated_mlp",),
    "swiglu": ("gated_mlp",),
    "embedding": ("embedding",),
    "embed": ("embedding",),
    "pos": ("positional",),
    "positional": ("positional",),
    "rope": ("rope",),
    "rotary": ("rope",),
    "rnn": ("rnn",),
    "lstm": ("rnn",),
    "gru": ("rnn",),
    "moe": ("moe",),
    "expert": ("moe",),
    "router": ("moe",),
    "ssm": ("ssm",),
    "mamba": ("ssm",),
    "rwkv": ("rwkv",),
    "gnn": ("gnn",),
    "graph": ("gnn",),
    "message": ("gnn",),
    "point": ("point_mlp",),
    "pointnet": ("point_mlp",),
    "nerf": ("nerf",),
    "vae": ("vae",),
    "reparam": ("vae",),
    "diffusion": ("diffusion",),
    "swin": ("swin",),
    "window": ("swin",),
    "lora": ("lora",),
    "patch": ("patch_embed",),
    "pool": ("pool",),
    "global": ("global_pool",),
    "upsample": ("upsample",),
    "concat": ("concat",),
    "add": ("add",),
    "weight": ("weight_tying",),
    "tying": ("weight_tying",),
}


#: Exact component ids shipped by ``netprism/data/algorithm_kb.json``.
_COMPONENT_SIGNAL_KEYS.update(
    {
        "plain_conv": ("conv",),
        "normalization": ("norm",),
        "token_embedding": ("embedding",),
        "classification_head": ("classifier_head",),
        "segmentation_head": ("segmentation_head",),
        "detection_head": ("detection_head",),
        "projection_head": ("projection_head", "classifier_head"),
        "rnn_block": ("rnn_block", "rnn"),
        "lstm_gate": ("lstm_gate", "rnn"),
        "gru_gate": ("gru_gate", "rnn"),
        "patch_embed": ("patch_embed",),
        "window_attention": ("swin",),
        "convnext_block": ("convnext_block",),
        "atrous_conv": ("atrous_conv", "conv"),
        "global_context": ("global_pool",),
        "concat_fusion": ("concat",),
        "upsampling": ("upsample", "deconv"),
        "timestep_embedding": ("timestep_embedding",),
        "alibi": ("alibi",),
        "kan_spline": ("kan_spline",),
        "capsule_routing": ("capsule_routing",),
        "neural_ode_solver": ("neural_ode_solver",),
        "vq_codebook": ("vq_codebook",),
        "weight_sharing": ("weight_tying",),
        "pafpn": ("pafpn", "panet"),
        "bifpn": ("bifpn", "fpn"),
        "depthwise_separable": ("depthwise_separable", "depthwise"),
        "dense_connection": ("dense_connection", "dense"),
        "unet_skip": ("unet_skip",),
        "swin": ("swin",),
    }
)

_TOKEN_SIGNAL_KEYS.update(
    {
        "plain": ("conv",),
        "projection": ("projection_head", "classifier_head"),
        "segmentation": ("segmentation_head",),
        "seg": ("segmentation_head",),
        "detection": ("detection_head",),
        "timestep": ("timestep_embedding",),
        "temb": ("timestep_embedding",),
        "alibi": ("alibi",),
        "convnext": ("convnext_block",),
        "atrous": ("atrous_conv",),
        "dilated": ("atrous_conv",),
        "capsule": ("capsule_routing",),
        "routing": ("capsule_routing",),
        "ode": ("neural_ode_solver",),
        "spline": ("kan_spline",),
        "kan": ("kan_spline",),
        "vq": ("vq_codebook",),
        "codebook": ("vq_codebook",),
        "sharing": ("weight_tying",),
        "upsampling": ("upsample",),
        "fusion": ("concat",),
        "windowattention": ("swin",),
    }
)


def _component_keys(component_id: Any) -> tuple[str, ...]:
    if not isinstance(component_id, str):
        return ()
    raw = component_id.strip().lower()
    if raw in _COMPONENT_SIGNAL_KEYS:
        return _COMPONENT_SIGNAL_KEYS[raw]
    cid = _norm_text(raw)
    if not cid:
        return ()
    if cid in _COMPONENT_SIGNAL_KEYS:
        return _COMPONENT_SIGNAL_KEYS[cid]
    keys: list[str] = []
    for token in re.split(r"[^0-9a-z]+", raw):
        if token in _TOKEN_SIGNAL_KEYS:
            for key in _TOKEN_SIGNAL_KEYS[token]:
                if key not in keys:
                    keys.append(key)
    return tuple(keys)


def _structure_facts(
    ctx: _Ctx, patterns: list[dict[str, Any]]
) -> tuple[dict[str, bool], dict[str, str]]:
    facts: dict[str, bool] = {}
    details: dict[str, str] = {}

    def mark(key: str, count: int, detail: str) -> None:
        if count > 0:
            facts[key] = True
            details[key] = detail

    for kind, count in ctx.kind_counts.items():
        label = _KIND_LABELS.get(kind, kind)
        suffix = f"×{count}" if count > 1 else ""
        mark(kind, count, f"{label}{suffix}")

    # normalization variants
    batchnorm = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "norm" and "batchnorm" in (_op_short(ctx.nodes[nid]).lower() + _norm_text(ctx.nodes[nid].get("name")))
    ]
    layernorm = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "norm" and "layernorm" in (_op_short(ctx.nodes[nid]).lower() + _norm_text(ctx.nodes[nid].get("name")))
    ]
    rmsnorm = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "norm" and "rms" in (_op_short(ctx.nodes[nid]).lower() + _norm_text(ctx.nodes[nid].get("name")))
    ]
    if batchnorm:
        mark("batchnorm", len(batchnorm), f"BatchNorm×{len(batchnorm)}")
    if layernorm:
        mark("layernorm", len(layernorm), f"LayerNorm×{len(layernorm)}")
    if rmsnorm:
        mark("rmsnorm", len(rmsnorm), f"RMSNorm×{len(rmsnorm)}")

    pattern_counts: dict[str, int] = {}
    for pattern in patterns:
        kind = str(pattern.get("kind") or "")
        pattern_counts[kind] = pattern_counts.get(kind, 0) + 1
    for kind, count in pattern_counts.items():
        label = PATTERN_SPECS.get(kind, {}).get("label", kind)
        suffix = f"×{count}" if count > 1 else ""
        mark(kind, count, f"{label}{suffix}")
        signal = _PATTERN_SIGNAL.get(kind)
        if signal:
            mark(signal, count, f"{label}{suffix}")
    if pattern_counts.get("residual_block") or pattern_counts.get("inverted_residual"):
        total = pattern_counts.get("residual_block", 0) + pattern_counts.get("inverted_residual", 0)
        mark("residual", total, f"残差结构×{total}")
    if pattern_counts.get("depthwise_separable") or ctx.kind_counts.get("conv", 0) and any(
        _is_depthwise(ctx.nodes[nid]) for nid in ctx.order
    ):
        mark("depthwise", max(1, pattern_counts.get("depthwise_separable", 0)), "深度卷积")
    if pattern_counts.get("gated_mlp"):
        tags = set()
        for pattern in patterns:
            if pattern.get("kind") == "gated_mlp":
                tags.update(pattern.get("tags") or [])
        if "swiglu" in tags:
            mark("swiglu", pattern_counts.get("gated_mlp", 0), "SwiGLU 门控 MLP")
    # gated MLP pattern tags always contain swiglu in the catalogue; expose it
    if pattern_counts.get("gated_mlp"):
        mark("swiglu", pattern_counts.get("gated_mlp", 0), "SwiGLU 门控 MLP")

    global_pools = [nid for nid in ctx.order if _is_global_pool(ctx, ctx.nodes[nid])]
    if global_pools:
        mark("global_pool", len(global_pools), f"全局池化×{len(global_pools)}")

    skip_edges = [
        edge for edge in ctx.edges if str(edge.get("kind") or "") in ("skip", "residual")
    ]
    skip_count = len(skip_edges) + pattern_counts.get("unet_skip", 0)
    if skip_count:
        mark("skip", skip_count, f"跳连×{skip_count}")

    positional = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "positional"
        or "posemb" in _norm_text(ctx.nodes[nid].get("name"))
        or "positional" in _norm_text(ctx.nodes[nid].get("name"))
    ]
    if positional:
        mark("positional", len(positional), f"位置编码×{len(positional)}")

    # op specific gates / dilated / modern conv blocks -------------------
    lstm_nodes = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "rnn" and "lstm" in _op_short(ctx.nodes[nid]).lower()
    ]
    gru_nodes = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "rnn" and "gru" in _op_short(ctx.nodes[nid]).lower()
    ]
    if lstm_nodes:
        mark("lstm_gate", len(lstm_nodes), f"LSTM 门控×{len(lstm_nodes)}")
    if gru_nodes:
        mark("gru_gate", len(gru_nodes), f"GRU 门控×{len(gru_nodes)}")
    if pattern_counts.get("rnn_block"):
        mark("rnn_block", pattern_counts.get("rnn_block", 0), f"循环块×{pattern_counts.get('rnn_block', 0)}")

    atrous = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "conv"
        and any(dim > 1 for dim in _as_int_list(_param(_params(ctx.nodes[nid]), "dilation", "dilations")))
    ]
    if atrous:
        mark("atrous_conv", len(atrous), f"空洞卷积×{len(atrous)}")

    depthwise_nodes = [nid for nid in ctx.order if _is_depthwise(ctx.nodes[nid])]
    large_depthwise = [
        nid
        for nid in depthwise_nodes
        if any(dim >= 7 for dim in _kernel_list(_params(ctx.nodes[nid])))
    ]
    if large_depthwise and (pattern_counts.get("residual_block") or pattern_counts.get("depthwise_separable")):
        mark("convnext_block", len(large_depthwise), f"7×7 Depthwise 残差块×{len(large_depthwise)}")

    # heads -------------------------------------------------------------
    output_signature = _model_output_shape(ctx)
    terminal_convs = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) == "conv"
        and not any(
            _node_kind(ctx.nodes[target]) in ("conv", "deconv", "attention", "linear")
            for target in ctx.succs.get(nid, [])
        )
    ]
    if terminal_convs and output_signature and len(output_signature) == 4:
        out_channels = output_signature[1]
        if out_channels is not None and out_channels <= 4 and (
            pattern_counts.get("unet_skip") or ctx.kind_counts.get("upsample", 0) or ctx.kind_counts.get("deconv", 0)
        ):
            mark("segmentation_head", len(terminal_convs), "末端逐像素分类卷积（分割头）")

    terminal_linears = [
        nid
        for nid in ctx.order
        if _node_kind(ctx.nodes[nid]) in ("linear", "mlp")
        and not any(_node_kind(ctx.nodes[target]) in ("linear", "mlp", "ffn") for target in ctx.succs.get(nid, []))
    ]
    tokens = _name_tokens(ctx.nodes.values(), ctx.model)
    normalised = " ".join(sorted(tokens))
    detection_hint = any(
        hint in normalised
        for hint in ("yolo", "detection", "anchor", "bbox", "fasterrcnn", "maskrcnn", "retinanet", "ssd")
    )
    if detection_hint and (ctx.kind_counts.get("reshape", 0) or ctx.kind_counts.get("custom", 0) or terminal_linears):
        mark("detection_head", max(1, len(terminal_linears)), "检测头（bbox/anchor）")
    if terminal_linears and (
        ctx.kind_counts.get("attention", 0) or ctx.kind_counts.get("embedding", 0) or detection_hint
    ):
        mark("projection_head", len(terminal_linears), "末端线性投影头")
    if any(
        hint in normalised
        for hint in ("timestep", "timeembed", "temb", "timeembedding", "film", "adaln")
    ):
        mark("timestep_embedding", 1, "时间步嵌入（FiLM/AdaGN）")

    # name based hints for non-mainstream architectures
    hint_tokens: dict[str, tuple[str, ...]] = {
        "rope": ("rope", "rotary", "rotaryemb"),
        "lora": ("lora", "adapter"),
        "moe": ("moe", "expert", "experts", "router", "mixtral", "switch"),
        "ssm": ("mamba", "ssm", "selectivescan", "statespace", "s4", "s6"),
        "rwkv": ("rwkv", "timemix", "receptance"),
        "gnn": ("gnn", "gcn", "gat", "graphsage", "graph", "messagepassing", "mpnn", "gin"),
        "point_mlp": ("pointnet", "pointmlp", "setabstraction", "pointcloud", "pointnet2"),
        "nerf": ("nerf", "radiance", "viewdir", "implicitfield", "instantngp"),
        "vae": ("vae", "variational", "reparameterization", "reparam", "vqvae", "codebook"),
        "diffusion": ("diffusion", "ddpm", "ddim", "denoiser", "dit"),
        "swin": ("swin", "windowattn", "shiftedwindow", "windowmsa"),
        "cbam": ("cbam",),
        "sk": ("sk", "sknet", "selectivekernel"),
        "fpn": ("fpn", "featurepyramid", "lateral"),
        "panet": ("panet",),
        "bifpn": ("bifpn",),
        "unet_plus_plus": ("unetplusplus", "unetplus", "nestedunet"),
        "patch_embed": ("patchembed", "patchembedding", "patchify"),
        "alibi": ("alibi",),
        "kan_spline": ("kan", "spline", "bspline"),
        "capsule_routing": ("capsule", "capsnet", "routing"),
        "neural_ode_solver": ("neuralode", "torchdiffeq", "odesolver", "continuousdepth"),
        "vq_codebook": ("vqvae", "vectorquant", "codebook"),
        "timestep_embedding": ("timestep", "timeembed", "temb", "film", "adaln"),
    }
    for key, hints in hint_tokens.items():
        if any(hint in normalised for hint in hints):
            mark(key, max(1, pattern_counts.get(key, 0)), f"名称命中 {hints[0]}")
    return facts, details


def _model_output_shape(ctx: _Ctx) -> list[int] | None:
    spec = ctx.model.get("outputSpec")
    if isinstance(spec, list) and spec and isinstance(spec[0], dict):
        shape = spec[0].get("shape")
        if isinstance(shape, list) and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
            return [int(dim) for dim in shape]
    for node_id in reversed(ctx.order):
        if _node_kind(ctx.nodes[node_id]) == "output":
            shape = _first_shape(ctx.nodes[node_id])
            if shape:
                return shape
    return None


def _component_evidence(
    ctx: _Ctx,
    facts: dict[str, bool],
    details: dict[str, str],
    component: dict[str, Any] | None,
    component_id: Any,
) -> tuple[float, str | None]:
    keys = _component_keys(component_id)
    if keys:
        hits = [key for key in keys if facts.get(key)]
        if hits:
            score = 0.62 + 0.38 * len(hits) / len(keys)
            reason = "、".join(details.get(key, key) for key in hits[:3])
            return round(score, 3), reason
        return 0.0, None
    if isinstance(component, dict):
        required_ops = [
            op for op in component.get("required_ops") or [] if isinstance(op, str) and op in _KIND_SET
        ]
        if required_ops:
            hits = [op for op in required_ops if ctx.kind_counts.get(op, 0) > 0]
            if hits:
                score = 0.3 + 0.25 * len(hits) / len(required_ops)
                return round(score, 3), "算子签名：" + "/".join(hits)
    return 0.0, None


def _ops_similarity(signature_ops: Any, kind_counts: dict[str, int]) -> float:
    if not isinstance(signature_ops, (list, tuple)):
        return 0.0
    signature = {op for op in signature_ops if isinstance(op, str) and op in _KIND_SET}
    if not signature:
        return 0.0
    graph_ops = {kind for kind, count in kind_counts.items() if count > 0 and kind in _KIND_SET}
    if not graph_ops:
        return 0.0
    intersection = len(signature & graph_ops)
    union = len(signature | graph_ops)
    jaccard = intersection / union if union else 0.0
    coverage = intersection / len(signature)
    return round(0.4 * jaccard + 0.6 * coverage, 4)


def _alias_score(
    names: list[str], family_id: str, family: dict[str, Any]
) -> tuple[float, str | None]:
    candidates: list[str] = []
    for value in [family_id] + list(family.get("aliases") or []):
        if not isinstance(value, str):
            continue
        normalised = _norm_text(value)
        if len(normalised) < 3 or normalised in _ALIAS_STOPWORDS:
            continue
        candidates.append(normalised)
    best = 0.0
    best_text: str | None = None
    for candidate in candidates:
        for name in names:
            if not name:
                continue
            if name == candidate:
                score = 3.6
            elif candidate in name:
                ratio = len(candidate) / max(1, len(name))
                score = 2.2 + 1.2 * min(1.0, ratio)
            elif name in candidate and len(name) >= 4:
                score = 1.6
            else:
                continue
            if score > best:
                best = score
                best_text = candidate
            break
    if best <= 0:
        return 0.0, None
    return best, f"名称/类名命中别名 {best_text!r}"


def _structure_summary(ctx: _Ctx) -> str:
    parts = []
    for kind in ("conv", "deconv", "linear", "attention", "embedding", "rnn", "pool", "norm", "activation", "add", "concat", "upsample"):
        count = ctx.kind_counts.get(kind, 0)
        if count:
            parts.append(f"{kind}={count}")
    input_rank = output_rank = None
    if isinstance(ctx.model.get("inputSpec"), list) and ctx.model["inputSpec"]:
        first = ctx.model["inputSpec"][0]
        if isinstance(first, dict):
            shape = first.get("shape")
            if isinstance(shape, list):
                input_rank = len(shape)
    if isinstance(ctx.model.get("outputSpec"), list) and ctx.model["outputSpec"]:
        first = ctx.model["outputSpec"][0]
        if isinstance(first, dict):
            shape = first.get("shape")
            if isinstance(shape, list):
                output_rank = len(shape)
    if input_rank or output_rank:
        parts.append(f"input_rank={input_rank}, output_rank={output_rank}")
    return "结构: " + (", ".join(parts) if parts else "空图")


def _build_candidates(
    ctx: _Ctx,
    facts: dict[str, bool],
    details: dict[str, str],
    patterns: list[dict[str, Any]],
    kb: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    families = [
        family
        for family in kb.get("families") or []
        if isinstance(family, dict) and isinstance(family.get("id"), str) and family.get("id") != "unknown"
    ]
    components = {
        component["id"]: component
        for component in kb.get("components") or []
        if isinstance(component, dict) and isinstance(component.get("id"), str)
    }
    names = _name_candidates(ctx.model)
    kind_counts = ctx.kind_counts
    raw_items: list[dict[str, Any]] = []
    for family in families:
        family_id = family["id"]
        label = str(family.get("label") or _FAMILY_LABELS.get(family_id) or family_id)
        score = 0.0
        evidence: list[str] = []

        alias_score, alias_reason = _alias_score(names, family_id, family)
        if alias_score:
            score += alias_score
            if alias_reason:
                evidence.append(alias_reason)

        required = [cid for cid in family.get("required_patterns") or [] if isinstance(cid, str)]
        required_scores: list[float] = []
        for component_id in required:
            component_score, reason = _component_evidence(
                ctx, facts, details, components.get(component_id), component_id
            )
            required_scores.append(component_score)
            if component_score >= 0.5 and reason:
                evidence.append(f"必需组件 {component_id}: {reason}")
        if required_scores:
            ratio = sum(min(1.0, item) for item in required_scores) / len(required_scores)
            score += 3.4 * ratio
            if all(item >= 0.6 for item in required_scores):
                score += 1.0
                evidence.append("必需组件全部满足")
        else:
            ratio = 0.0

        optional = [cid for cid in family.get("optional_patterns") or [] if isinstance(cid, str)]
        optional_scores: list[float] = []
        for component_id in optional:
            component_score, reason = _component_evidence(
                ctx, facts, details, components.get(component_id), component_id
            )
            optional_scores.append(component_score)
            if component_score >= 0.6 and reason:
                evidence.append(f"可选组件 {component_id}: {reason}")
        if optional_scores:
            score += 1.1 * (sum(optional_scores) / len(optional_scores))

        similarity = _ops_similarity(family.get("signature_ops"), kind_counts)
        if similarity:
            score += 2.2 * similarity
            signature_ops = [
                op for op in family.get("signature_ops") or [] if isinstance(op, str) and op in _KIND_SET
            ]
            matched = sum(1 for op in signature_ops if kind_counts.get(op, 0) > 0)
            evidence.append(f"签名算子匹配 {matched}/{len(signature_ops)}")

        # SPEC canonical families stay representable; structural bonus only.
        if family_id in CANONICAL_FAMILIES and ratio >= 0.6:
            score += 1.4
            if family_id == "transformer_decoder":
                if _has_causal_mask(ctx):
                    score += 1.8
                    evidence.append("检测到因果掩码（decoder-only）")
            elif family_id == "transformer_encoder":
                if not _has_causal_mask(ctx) and _has_token_head(ctx):
                    score += 0.8
                    evidence.append("无因果掩码 + 分类头（encoder 结构）")

        if ratio < 0.5 and required:
            missing = [
                component_id
                for component_id, component_score in zip(required, required_scores)
                if component_score < 0.5
            ]
            if missing:
                evidence.append("缺必需组件: " + ", ".join(missing[:3]))

        # Generic KB fallbacks must stay behind more specific families unless
        # the model name itself points at them.
        if family_id in ("cnn", "mlp", "hybrid") and not alias_score:
            score -= 0.9
        # Families whose entire "required" evidence is generic plumbing
        # (normalization / plain conv / generic ffn) are weak candidates.
        generic_required = {"normalization", "plain_conv", "ffn", "conv_bn_act"}
        if required and set(required) <= generic_required and not alias_score:
            score -= 1.0

        raw_items.append(
            {
                "id": family_id,
                "label": label,
                "score": score,
                "domain": family.get("domain"),
                "alias": alias_score,
                "required_full": bool(required_scores) and all(
                    item >= 0.6 for item in required_scores
                ),
                "evidence": evidence,
            }
        )

    raw_items.sort(key=lambda item: (-item["score"], item["id"]))
    raw_items = _promote_canonical(raw_items, facts)
    positive = [item for item in raw_items if item["score"] > 0.05]
    if not positive:
        return [], "unknown"
    top_score = positive[0]["score"]
    pool = positive[:6]
    tau = max(0.9, 0.22 * top_score)
    weights = [math.exp((item["score"] - top_score) / tau) for item in pool]
    total = sum(weights) or 1.0

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(pool[:3]):
        absolute = 0.94 * item["score"] / (item["score"] + 2.6)
        relative = weights[index] / total
        confidence = round(min(0.98, max(0.02, 0.62 * absolute + 0.38 * relative)), 2)
        # Calibration: an exact/near-exact name match with every required
        # component evidenced is a high-confidence family classification.
        if index == 0 and item.get("alias", 0) >= 3.0 and item.get("required_full"):
            confidence = max(confidence, 0.78)
        if index == 0 and item.get("promoted"):
            confidence = max(confidence, 0.72)
        candidates.append(
            {
                "id": item["id"],
                "label": item["label"],
                "confidence": confidence,
                "evidence": item["evidence"][:5],
            }
        )

    if not candidates or candidates[0]["confidence"] < 0.35:
        # No family is well supported: the caller reports ``unknown``.  Keep
        # only genuine (weak) candidates here — ``unknown`` is not a family.
        return candidates, "unknown"
    return candidates, None


#: Families that are specialisations of SPEC-ALGO canonical ids.  The
#: canonical id stays representable for legacy CLI/report consumers while the
#: KB-specific family remains in the candidate list.
_SPEC_COMPAT_FAMILIES: dict[str, str] = {
    "gpt": "transformer_decoder",
    "bert": "transformer_encoder",
}

_TRANSFORMER_CORE = ("transformer_block", "self_attention", "ffn")


def _promote_canonical(
    items: list[dict[str, Any]], facts: dict[str, bool]
) -> list[dict[str, Any]]:
    """Place the SPEC canonical family first when a generic LLM family wins.

    ``gpt``/``bert`` are KB refinements of ``transformer_decoder`` /
    ``transformer_encoder``.  For backward compatibility (SPEC-ALGO enums)
    the canonical id is promoted when the shared transformer core is fully
    evidenced; the KB family stays as the next candidate.
    """
    if not items:
        return items
    top = items[0]
    target_id = _SPEC_COMPAT_FAMILIES.get(str(top.get("id")))
    if not target_id:
        return items
    if not all(facts.get(signal) for signal in _TRANSFORMER_CORE):
        return items
    target = next((item for item in items if item.get("id") == target_id), None)
    if target is None:
        return items
    promoted = dict(target)
    promoted["score"] = max(float(promoted.get("score") or 0), float(top.get("score") or 0) + 0.02)
    promoted["promoted"] = True
    promoted["evidence"] = [
        f"SPEC 兼容家族：{top.get('id')} 的通用 Transformer 结构归入 {target_id}"
    ] + list(promoted.get("evidence") or [])
    return [promoted] + [item for item in items if item.get("id") != target_id]


def _has_causal_mask(ctx: _Ctx) -> bool:
    for node_id in ctx.order:
        node = ctx.nodes[node_id]
        text = _norm_text(node.get("name")) + " " + _norm_text(node.get("modulePath"))
        if "mask" in text and ("causal" in text or "attn" in text or _node_kind(node) == "attention"):
            return True
        if _node_kind(node) == "attention" and "maskedfill" in _norm_text(_op_short(node)):
            return True
    return False


def _has_token_head(ctx: _Ctx) -> bool:
    if ctx.kind_counts.get("embedding", 0) > 0:
        return True
    return any(_node_kind(ctx.nodes[nid]) in ("linear", "mlp") for nid in ctx.order)


def _unknown_reason(
    ctx: _Ctx, families: list[dict[str, Any]], patterns: list[dict[str, Any]]
) -> str | None:
    if not families:
        if not ctx.nodes:
            return "空图：没有可分析的节点"
        pattern_kinds = sorted({str(pattern.get("kind")) for pattern in patterns})
        return "；".join(
            [
                "结构证据不足，未能匹配任何家族签名",
                _structure_summary(ctx),
                "已识别模式: " + ("、".join(pattern_kinds) if pattern_kinds else "无"),
            ]
        )
    top = families[0]
    if top.get("id") != "unknown" and float(top.get("confidence") or 0) >= 0.35:
        return None
    pattern_kinds = sorted({str(pattern.get("kind")) for pattern in patterns})
    return "；".join(
        [
            "结构证据不足，未能稳定匹配家族",
            _structure_summary(ctx),
            "已识别模式: " + ("、".join(pattern_kinds) if pattern_kinds else "无"),
        ]
    )


# --------------------------------------------------------------------------- #
# algorithm tags
# --------------------------------------------------------------------------- #
_FAMILY_TAGS: dict[str, list[str]] = {
    "cnn": ["cnn"],
    "vgg": ["cnn", "vgg"],
    "resnet": ["cnn", "residual"],
    "densenet": ["cnn", "dense"],
    "mobilenet": ["cnn", "depthwise"],
    "mobilenet_v2": ["cnn", "depthwise", "inverted_residual"],
    "inception": ["cnn", "multi_branch"],
    "unet": ["cnn", "segmentation", "skip_connection"],
    "transformer_encoder": ["transformer", "encoder", "attention"],
    "transformer_decoder": ["transformer", "decoder", "attention"],
    "vit": ["transformer", "vision", "patch_embedding"],
    "mlp": ["mlp"],
    "rnn": ["rnn", "sequence"],
    "hybrid": ["hybrid"],
}


def _algorithm_tags(
    ctx: _Ctx,
    families: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
) -> list[str]:
    tags: list[str] = []
    family_id = str(families[0].get("id")) if families else ""
    for tag in _FAMILY_TAGS.get(family_id, []):
        if tag not in tags:
            tags.append(tag)
    if family_id and family_id != "unknown" and family_id not in tags:
        tags.append(family_id)
    for pattern in patterns:
        for tag in pattern.get("tags") or []:
            if isinstance(tag, str) and tag and tag not in tags:
                tags.append(tag)
    return tags[:16]


# --------------------------------------------------------------------------- #
# forward pseudocode
# --------------------------------------------------------------------------- #
def _shape_text(shape: list[int] | None) -> str:
    if not shape:
        return "?"
    parts = ["?" if dim < 0 else str(dim) for dim in shape]
    return "(" + ", ".join(parts) + ")"


def _var_name(node_id: str) -> str:
    if node_id.startswith("n") and node_id[1:].isdigit():
        return f"v{node_id[1:]}"
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", node_id).strip("_") or "v"
    return f"v{cleaned}"


def _node_call(node: dict[str, Any]) -> str:
    kind = _node_kind(node)
    op = _op_short(node)
    params = _params(node)
    if kind == "conv":
        if _is_depthwise(node):
            name = "DepthwiseConv2d" if _rank(node) == 4 else "DepthwiseConv1d"
        else:
            name = "Conv2d" if _rank(node) != 3 else "Conv1d"
            if _rank(node) == 5:
                name = "Conv3d"
        out_channels = _int_param(params, "outChannels", "out_channels")
        args = [str(out_channels or "?"), _kernel_text(params)]
        stride = _as_int_list(_param(params, "stride", "strides"))
        if stride and any(dim != 1 for dim in stride):
            args.append("s" + "x".join(str(dim) for dim in stride))
        padding = _as_int_list(_param(params, "padding"))
        if padding and any(dim != 0 for dim in padding):
            args.append("p" + "x".join(str(dim) for dim in padding))
        groups = _int_param(params, "groups", default=1)
        if groups > 1 and not _is_depthwise(node):
            args.append(f"g{groups}")
        return f"{name}({','.join(args)})"
    if kind == "deconv":
        name = "ConvTranspose2d" if _rank(node) == 4 else "ConvTranspose1d"
        out_channels = _int_param(params, "outChannels", "out_channels")
        args = [str(out_channels or "?"), _kernel_text(params)]
        stride = _as_int_list(_param(params, "stride", "strides"))
        if stride and any(dim != 1 for dim in stride):
            args.append("s" + "x".join(str(dim) for dim in stride))
        return f"{name}({','.join(args)})"
    if kind == "linear":
        out_features = _int_param(params, "outFeatures", "out_features")
        return f"Linear({out_features or '?'})"
    if kind == "norm":
        short = op.lower()
        channels = _int_param(params, "normalizedShape", "num_features", default=0)
        if not channels:
            shape = _as_int_list(_param(params, "normalizedShape"))
            channels = shape[-1] if shape else 0
        if "batchnorm" in short:
            return f"BatchNorm({channels or '?'})"
        if "rms" in short:
            return f"RMSNorm({channels or '?'})"
        if "group" in short:
            return f"GroupNorm({channels or '?'})"
        return f"LayerNorm({channels or '?'})"
    if kind == "activation":
        return f"{op or 'Act'}()"
    if kind == "softmax":
        dim = _as_int(_param(params, "dim"))
        return f"Softmax({dim if dim is not None else -1})"
    if kind == "pool":
        if _is_global_pool(_noop_ctx(node), node) or _is_global_max_pool(_noop_ctx(node), node):
            return "GlobalMaxPool()" if "max" in op.lower() else "GlobalAvgPool()"
        ptype = str(_param(params, "type") or "").lower()
        name = "MaxPool" if ("max" in op.lower() or ptype == "max") else "AvgPool"
        kernel = _kernel_text(params)
        stride = _as_int_list(_param(params, "stride", "strides"))
        args = [kernel]
        if stride and any(dim != 1 for dim in stride):
            args.append("s" + "x".join(str(dim) for dim in stride))
        return f"{name}({','.join(args)})"
    if kind == "dropout":
        p = _param(params, "p")
        return f"Dropout({p})" if p is not None else "Dropout()"
    if kind == "embedding":
        vocab = _int_param(params, "numEmbeddings", "num_embeddings")
        dim = _int_param(params, "embeddingDim", "embedding_dim")
        node_text = _norm_text(node.get("name")) + " " + _norm_text(node.get("modulePath"))
        call = "PositionalEmbedding" if ("pos" in node_text or "position" in node_text) else "Embedding"
        return f"{call}({vocab or '?'},{dim or '?'})"
    if kind == "positional":
        return "PositionalEncoding()"
    if kind == "attention":
        if "multihead" in op.lower() or _param(params, "numHeads"):
            heads = _as_int(_param(params, "numHeads"))
            return f"MultiHeadAttention({heads or '?'}h)"
        if "masked" in op.lower():
            return "MaskedFill()"
        if "matmul" in op.lower() or "bmm" in op.lower():
            return "MatMul()"
        return "Attention()"
    if kind == "ffn":
        hidden = _int_param(params, "hiddenFeatures", "dimFeedforward")
        return f"FeedForward({hidden or '?'})"
    if kind == "mlp":
        return "MLP()"
    if kind == "rnn":
        hidden = _int_param(params, "hiddenFeatures", "hidden_size")
        return f"{op or 'RNN'}({hidden or '?'})"
    if kind == "upsample":
        scale = _param(params, "scaleFactor")
        if scale is not None:
            return f"Upsample(x{scale})"
        return "Upsample()"
    if kind == "flatten":
        return "Flatten()"
    if kind == "reshape":
        return f"{op or 'Reshape'}()"
    if kind == "block":
        return f"{op or 'Block'}()"
    if kind == "container":
        return "Container()"
    if kind == "custom":
        return f"{_call_name(op or str(node.get('name') or 'Custom'))}()"
    if kind in ("input", "output"):
        return kind.title()
    return f"{_call_name(op or 'Op')}()"


def _call_name(op: str) -> str:
    name = str(op or "Op").rstrip("_")
    if not name:
        return "Op"
    return name[:1].upper() + name[1:]


class _NoopCtx:
    def __init__(self, node: dict[str, Any]) -> None:
        self.node = node

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive
        return None


def _noop_ctx(node: dict[str, Any]) -> Any:
    """Small shim so ``_is_global_pool`` can be reused without a full graph."""
    class _Shim:
        nodes = {node.get("id", "n0"): node}

    return _Shim()


def _feature_size(ctx: _Ctx, node_id: str, kind: str) -> int | None:
    shape = _first_shape(ctx.nodes[node_id])
    if not shape:
        return None
    if kind in ("transformer_block", "attention_block", "ffn_block", "rnn_block", "gated_mlp", "moe"):
        value = shape[-1]
        return value if value > 0 else None
    if len(shape) == 4:
        value = shape[1]
        return value if value > 0 else None
    if len(shape) == 3:
        value = shape[1]
        return value if value > 0 else None
    value = shape[-1]
    return value if value > 0 else None


def _block_call(ctx: _Ctx, group: dict[str, Any], pattern: dict[str, Any] | None) -> str:
    kind = str(group.get("kind") or "")
    pattern_kind = str(pattern.get("kind") or "") if pattern else ""
    if pattern_kind in _BLOCK_NAMES:
        name = _BLOCK_NAMES[pattern_kind]
    else:
        name = {
            "conv_block": "ConvBlock",
            "residual_block": "ResidualBlock",
            "transformer_block": "TransformerBlock",
            "attention_block": "AttentionBlock",
            "ffn_block": "FeedForward",
            "unet_block": "UNetBlock",
            "mlp": "MLP",
        }.get(kind, "Block")
    last_name = str(group.get("name") or "").lower()
    if name == "UNetBlock":
        last_part = last_name.rsplit(".", 1)[-1]
        if last_part.startswith(("enc", "down")):
            name = "EncoderBlock"
        elif last_part.startswith(("dec", "up")):
            name = "DecoderBlock"
    subtree = ctx.subtree_nodes(group["id"])
    if subtree:
        size = _feature_size(ctx, subtree[-1], pattern_kind or kind)
        if size:
            return f"{name}({size})"
    return f"{name}()"


def _group_items(ctx: _Ctx, group_id: str) -> list[tuple[int, int, str]]:
    items: list[tuple[int, int, str]] = []
    for node_id in ctx.group_direct.get(group_id, []):
        items.append((ctx.pos.get(node_id, 10**9), 0, node_id))
    for child in ctx.group_children.get(group_id, []):
        subtree = ctx.subtree_nodes(child)
        if subtree:
            items.append((min(ctx.pos.get(nid, 10**9) for nid in subtree), 1, child))
    items.sort()
    return items


def _is_foldable(ctx: _Ctx, group_id: str, group_patterns: dict[str, dict[str, Any]]) -> bool:
    if group_id == ctx.root_group_id:
        return False
    group = ctx.group_by_id.get(group_id)
    if group is None:
        return False
    subtree = ctx.subtree_nodes(group_id)
    if len(subtree) < 2:
        return False
    kind = str(group.get("kind") or "")
    pattern = group_patterns.get(group_id)
    has_foldable_pattern = pattern is not None and pattern.get("kind") in _FOLDABLE_PATTERN_KINDS
    if kind not in _FOLD_KINDS and not has_foldable_pattern:
        return False
    if any(_node_kind(ctx.nodes[nid]) in ("input", "output") for nid in subtree):
        return False
    # A group covering the whole network is the model itself, not a block:
    # keep its layers expanded so the pseudocode stays informative.
    computational = [
        nid for nid in ctx.order if _node_kind(ctx.nodes[nid]) not in ("input", "output")
    ]
    if len(subtree) >= 2 and set(subtree) == set(computational):
        return False
    # Multi-stage CNN containers (VGG ``features``: several pools) fold badly
    # into one line; expand them so every stage keeps its own shape.
    pool_nodes = [nid for nid in subtree if _node_kind(ctx.nodes[nid]) == "pool"]
    if len(pool_nodes) >= 2:
        return False
    positions = [ctx.pos.get(nid, -1) for nid in subtree]
    if any(position < 0 for position in positions):
        return False
    if max(positions) - min(positions) + 1 != len(positions):
        return False
    subtree_set = set(subtree)
    sinks = [
        nid
        for nid in subtree
        if not any(target in subtree_set for target in ctx.succs.get(nid, []))
    ]
    if not sinks:
        return False
    for node_id in subtree:
        for target in ctx.succs.get(node_id, []):
            if target not in subtree_set and node_id not in sinks:
                return False
    entries = [
        nid
        for nid in subtree
        if not any(source in subtree_set for source in ctx.preds.get(nid, []))
    ]
    if not entries:
        return False
    external_sources: list[str] = []
    for node_id in entries:
        for source in ctx.preds.get(node_id, []):
            if source in ctx.nodes and source not in subtree_set and source not in external_sources:
                external_sources.append(source)
    # A residual block has two entries (main path + shortcut) but a single
    # external input value, which still folds into a single block call.
    if len(external_sources) > 1:
        return False
    return True


def _build_pseudocode_plan(
    ctx: _Ctx, group_patterns: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    emitted: set[str] = set()

    def emit_node(node_id: str) -> None:
        if node_id in emitted:
            return
        emitted.add(node_id)
        plan.append({"type": "node", "node": node_id, "sink": node_id})

    def emit_block(group_id: str) -> None:
        subtree = ctx.subtree_nodes(group_id)
        if not subtree:
            return
        subtree_set = set(subtree)
        for node_id in subtree:
            emitted.add(node_id)
        entries = [
            nid
            for nid in subtree
            if not any(source in subtree_set for source in ctx.preds.get(nid, []))
        ]
        external_sources: list[str] = []
        for node_id in entries:
            for source in ctx.preds.get(node_id, []):
                if source in ctx.nodes and source not in subtree_set and source not in external_sources:
                    external_sources.append(source)
        plan.append(
            {
                "type": "block",
                "group": group_id,
                "entry": entries[0] if entries else subtree[0],
                "inputs": external_sources,
                "sink": subtree[-1],
                "nodes": subtree,
            }
        )

    def walk_items(items: Iterable[tuple[int, int, str]]) -> None:
        for _, item_type, item in items:
            if item_type == 0:
                if _node_kind(ctx.nodes[item]) != "output":
                    emit_node(item)
                continue
            if _is_foldable(ctx, item, group_patterns):
                emit_block(item)
            else:
                walk_items(_group_items(ctx, item))

    # Top level: roots' items plus nodes that live outside every group
    # (typically Input/Output boundaries), merged in execution order.
    top_items: list[tuple[int, int, str]] = []
    positioned: set[str] = set()
    for root_id in ctx.root_ids:
        for item in _group_items(ctx, root_id):
            top_items.append(item)
            if item[1] == 0:
                positioned.add(item[2])
    for node_id in ctx.order:
        if node_id in positioned or node_id in emitted:
            continue
        if node_id in ctx.node_group:
            continue
        if _node_kind(ctx.nodes[node_id]) == "output":
            continue
        top_items.append((ctx.pos.get(node_id, 10**9), 0, node_id))
    top_items.sort()
    walk_items(top_items)
    # nodes without a usable group still appear in execution order
    for node_id in ctx.order:
        if node_id in emitted:
            continue
        if _node_kind(ctx.nodes[node_id]) == "output":
            continue
        emit_node(node_id)
    return plan


def _plan_inputs(ctx: _Ctx, plan_item: dict[str, Any], group_patterns: dict[str, dict[str, Any]]) -> list[str]:
    if plan_item["type"] == "node":
        return [source for source in ctx.preds.get(plan_item["node"], [])]
    inputs = plan_item.get("inputs")
    if isinstance(inputs, list):
        return [source for source in inputs if source in ctx.nodes]
    entry = plan_item.get("entry")
    if not entry:
        return []
    subtree = set(plan_item.get("nodes") or [])
    return [source for source in ctx.preds.get(entry, []) if source not in subtree]


def _pseudocode(
    ctx: _Ctx,
    patterns: list[dict[str, Any]],
    group_patterns: dict[str, dict[str, Any]],
) -> list[str]:
    if not ctx.nodes:
        return []
    plan = _build_pseudocode_plan(ctx, group_patterns)
    if not plan:
        return []

    side_vars: set[str] = set()
    previous_sink: str | None = None
    for item in plan:
        for source in _plan_inputs(ctx, item, group_patterns):
            if source != previous_sink:
                side_vars.add(source)
        previous_sink = item.get("sink")

    def ref(source: str) -> str:
        if source in side_vars:
            return _var_name(source)
        return "x"

    def assign(sink: str) -> str:
        return _var_name(sink) if sink in side_vars else "x"

    lines: list[str] = []
    last_var = "x"
    index = 0
    while index < len(plan):
        item = plan[index]
        index += 1
        if len(lines) >= MAX_PSEUDOCODE_LINES:
            lines.append("# ...（伪代码已截断）")
            break
        if item["type"] == "node":
            node_id = item["node"]
            node = ctx.nodes[node_id]
            kind = _node_kind(node)
            if kind == "output":
                continue
            sources = [source for source in ctx.preds.get(node_id, []) if source in ctx.nodes]
            args = ", ".join(ref(source) for source in sources)
            variable = assign(node_id)
            shape = _shape_text(_display_shape(ctx, node_id))
            if kind == "input":
                input_shape = _first_shape(node) or _first_shape(node, "inShapes")
                lines.append(f"{variable} = Input({', '.join(str(dim) for dim in input_shape or ['?'])})")
                last_var = variable
                continue
            # embedding scale + positional add collapses to one readable line
            if (
                kind == "custom"
                and _is_mul(node)
                and len(sources) == 1
                and _node_kind(ctx.nodes[sources[0]]) == "embedding"
                and index < len(plan)
                and plan[index].get("type") == "node"
            ):
                next_id = plan[index]["node"]
                next_node = ctx.nodes[next_id]
                if _node_kind(next_node) == "add":
                    next_sources = [src for src in ctx.preds.get(next_id, []) if src in ctx.nodes]
                    others = [src for src in next_sources if src != node_id]
                    if not others or all(
                        _node_kind(ctx.nodes[src]) in ("embedding", "positional") for src in others
                    ):
                        pe_shape = _display_shape(ctx, next_id)
                        dim = pe_shape[-1] if pe_shape else None
                        variable = assign(next_id)
                        lines.append(
                            f"{variable} = PositionalEncoding({dim or '?'})({ref(sources[0])})"
                            f"   # -> {_shape_text(pe_shape)}"
                        )
                        last_var = variable
                        index += 1
                        continue
            if kind == "add" and len(sources) >= 2:
                expression = " + ".join(ref(source) for source in sources)
                line = f"{variable} = {expression}"
            elif kind == "custom" and _is_mul(node) and len(sources) >= 2:
                line = f"{variable} = {ref(sources[0])} * {ref(sources[1])}"
            elif kind == "concat":
                dim = _param(_params(node), "dim")
                dim_text = dim if dim is not None else 1
                line = f"{variable} = Concat(dim={dim_text})({args})"
            else:
                call = _node_call(node)
                if kind == "linear" and _is_terminal(ctx, node_id):
                    variable = "logits"
                line = f"{variable} = {call}({args})" if args else f"{variable} = {call}()"
            lines.append(f"{line}   # -> {shape}")
            last_var = variable
        else:
            group_id = item["group"]
            group = ctx.group_by_id[group_id]
            pattern = group_patterns.get(group_id)
            call = _block_call(ctx, group, pattern)
            sources = [source for source in _plan_inputs(ctx, item, group_patterns) if source in ctx.nodes]
            args = ", ".join(ref(source) for source in sources) if sources else "x"
            variable = assign(item["sink"])
            shape = _shape_text(_display_shape(ctx, item["sink"]))
            lines.append(f"{variable} = {call}({args})   # -> {shape}")
            last_var = variable
    if last_var == "logits":
        lines.append("return logits")
    return lines


def _display_shape(ctx: _Ctx, node_id: str) -> list[int] | None:
    """Node out-shape with sequence/batch fixes where the tracer is lossy."""
    node = ctx.nodes.get(node_id)
    if node is None:
        return None
    shape = _first_shape(node)
    if shape is None:
        return None
    if _node_kind(node) == "embedding" and len(shape) == 2:
        input_shape = _first_shape(node, "inShapes")
        if input_shape and len(input_shape) == 2 and shape[0] == input_shape[-1]:
            # traced positional embedding reports (seq, dim); the batch dim
            # comes from the token input, not from invention.
            return [input_shape[0], shape[0], shape[1]]
        model_shape = _model_input_shape(ctx)
        if model_shape and len(model_shape) >= 2 and shape[0] == model_shape[-1]:
            return [model_shape[0], shape[0], shape[1]]
    return shape


def _model_input_shape(ctx: _Ctx) -> list[int] | None:
    spec = ctx.model.get("inputSpec")
    if isinstance(spec, list) and spec and isinstance(spec[0], dict):
        shape = spec[0].get("shape")
        if isinstance(shape, list) and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
            return [int(dim) for dim in shape]
    for node_id in ctx.order:
        if _node_kind(ctx.nodes[node_id]) == "input":
            shape = _first_shape(ctx.nodes[node_id])
            if shape:
                return shape
    return None


def _is_terminal(ctx: _Ctx, node_id: str) -> bool:
    successors = ctx.succs.get(node_id, [])
    if not successors:
        return True
    return all(_node_kind(ctx.nodes[target]) == "output" for target in successors)


# --------------------------------------------------------------------------- #
# node formulas
# --------------------------------------------------------------------------- #
def _conv_axes(node: dict[str, Any]) -> str:
    rank = _rank(node)
    if rank == 3:
        return "NCW"
    if rank == 5:
        return "NCDHW"
    return "NCHW"


def _conv_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    cin = _int_param(params, "inChannels", "in_channels")
    cout = _int_param(params, "outChannels", "out_channels")
    groups = _int_param(params, "groups", default=1) or 1
    kernel = _kernel_text(params)
    stride = _as_int_list(_param(params, "stride", "strides"))
    padding = _as_int_list(_param(params, "padding"))
    rank = _rank(node)
    if rank == 3:
        index = "l"
        shift = f"l·s{'-p' if any(padding) else ''}+k"
        formula = (
            f"y[n,o,{index}'] = Σ_{{c,k}} W[o,c,k]·x[n,c,{shift}] + b[o]"
        )
        axes = f"NCW·{index}"
    elif rank == 5:
        formula = (
            "y[n,o,d',h',w'] = Σ_{c,kd,kh,kw} W[o,c,kd,kh,kw]·x[n,c,d,h,w] + b[o]"
        )
        axes = "NCDHW"
    else:
        formula = (
            "y[n,o,h',w'] = Σ_{c,kh,kw} W[o,c,kh,kw]·x[n,c,h·s+kh-p,w·s+kw-p] + b[o]"
        )
        axes = "NCHW"
    if _is_depthwise(node):
        formula = (
            "y[n,c,h',w'] = Σ_{kh,kw} W[c,kh,kw]·x[n,c,h·s+kh-p,w·s+kw-p] + b[c]"
        )
        semantic = "逐通道空间卷积（depthwise，groups=Cin）"
    elif all(dim == 1 for dim in _kernel_list(params)):
        semantic = "1×1 通道投影（逐像素线性混合）"
    elif groups > 1:
        semantic = f"分组空间卷积（groups={groups}，通道组内共享）"
        formula = (
            "y[n,o,h',w'] = Σ_{c∈g(o),kh,kw} W[o,c,kh,kw]·x[n,c,h·s+kh-p,w·s+kw-p] + b[o]"
        )
    elif any(dim >= 5 for dim in _kernel_list(params)):
        semantic = f"{kernel} 大核空间卷积（扩大感受野）"
    elif stride and any(dim > 1 for dim in stride):
        semantic = f"{kernel} 步长卷积（空间下采样）"
    else:
        semantic = f"{kernel} 空间卷积（通道 {cin}→{cout}）"
    suffix = f"  (Cin={cin or '?'}→Cout={cout or '?'}, k={kernel}, s={_int_list_text(stride)}, p={_int_list_text(padding)}"
    if groups > 1:
        suffix += f", groups={groups}"
    suffix += ")"
    tags = ["conv", "spatial"]
    if _is_depthwise(node):
        tags.append("depthwise")
    if all(dim == 1 for dim in _kernel_list(params)):
        tags.append("pointwise")
    if groups > 1:
        tags.append("grouped")
    if stride and any(dim > 1 for dim in stride):
        tags.append("downsample")
    return {"formula": formula + suffix, "semantic": semantic, "axes": axes, "tags": tags}


def _int_list_text(values: list[int]) -> str:
    if not values:
        return "?"
    return "x".join(str(dim) for dim in values)


def _deconv_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    cin = _int_param(params, "inChannels", "in_channels")
    cout = _int_param(params, "outChannels", "out_channels")
    kernel = _kernel_text(params)
    stride = _as_int_list(_param(params, "stride", "strides"))
    formula = (
        "y[n,o,h',w'] = Σ_{c,kh,kw} W[c,o,kh,kw]·x[n,c,h,w]"
        "  (h'=(h-1)·s-2p+k+op)"
    )
    semantic = "转置卷积上采样（学习式放大空间分辨率）"
    suffix = f"  (Cin={cin or '?'}→Cout={cout or '?'}, k={kernel}, s={_int_list_text(stride)})"
    return {
        "formula": formula + suffix,
        "semantic": semantic,
        "axes": "NCHW",
        "tags": ["deconv", "upsample", "spatial"],
    }


def _linear_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    in_features = _int_param(params, "inFeatures", "in_features")
    out_features = _int_param(params, "outFeatures", "out_features")
    bias = " + b[o]" if params.get("bias") is not False else ""
    formula = f"y[..., o] = Σ_i x[..., i]·W[o, i]{bias}"
    suffix = f"  (in={in_features or '?'}→out={out_features or '?'})"
    return {
        "formula": formula + suffix,
        "semantic": "全连接通道混合（特征维线性变换）",
        "axes": "N×F / NLC",
        "tags": ["linear", "channel_mix"],
    }


def _norm_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    op = _op_short(node).lower()
    eps = _param(params, "eps")
    eps_text = f", ε={eps}" if eps is not None else ""
    features = _int_param(params, "normalizedShape", "num_features")
    if not features:
        shape = _as_int_list(_param(params, "normalizedShape"))
        features = shape[-1] if shape else 0
    if "batchnorm" in op or "syncbatchnorm" in op:
        formula = (
            "y[n,c,:] = γ[c]·(x[n,c,:]-μ_B[c])/√(σ²_B[c]+ε) + β[c]"
        )
        return {
            "formula": formula + f"  (C={features or '?'}{eps_text})",
            "semantic": "批归一化：按通道统计量标准化并仿射",
            "axes": "NCHW（逐通道）",
            "tags": ["norm", "batchnorm"],
        }
    if "layernorm" in op:
        formula = "y[..., d] = γ[d]·(x[..., d]-μ)/√(σ²+ε) + β[d]"
        return {
            "formula": formula + f"  (E={features or '?'}{eps_text})",
            "semantic": "层归一化：对每个样本最后一维标准化",
            "axes": "最后一维 E",
            "tags": ["norm", "layernorm"],
        }
    if "rms" in op or "rms" in _norm_text(node.get("name")):
        formula = "y[..., d] = γ[d]·x[..., d]/√(mean_d(x²)+ε)"
        return {
            "formula": formula + f"  (E={features or '?'}{eps_text})",
            "semantic": "RMSNorm：仅用均方根缩放，无均值中心化",
            "axes": "最后一维 E",
            "tags": ["norm", "rmsnorm"],
        }
    if "group" in op:
        groups = _int_param(params, "numGroups", "num_groups", default=0)
        formula = "y = γ·(x-μ_group)/√(σ²_group+ε) + β"
        return {
            "formula": formula + f"  (groups={groups or '?'}, C={features or '?'})",
            "semantic": "分组归一化：组内统计量标准化",
            "axes": "NCHW（组内）",
            "tags": ["norm", "groupnorm"],
        }
    formula = "y = γ·(x-μ)/√(σ²+ε) + β"
    return {
        "formula": formula + f"  (features={features or '?'}{eps_text})",
        "semantic": "归一化层：标准化特征分布后仿射变换",
        "axes": None,
        "tags": ["norm"],
    }


_ACTIVATION_FORMULAS: dict[str, tuple[str, str]] = {
    "relu": ("y = max(x, 0)", "逐元素 ReLU 非线性（负值截断）"),
    "relu6": ("y = min(max(x, 0), 6)", "ReLU6：输出限制在 [0,6]"),
    "gelu": ("y = x·Φ(x)", "GELU：高斯误差线性单元"),
    "silu": ("y = x·σ(x)", "SiLU/Swish：自门控平滑激活"),
    "swish": ("y = x·σ(x)", "SiLU/Swish：自门控平滑激活"),
    "sigmoid": ("y = 1/(1+e^{-x})", "Sigmoid：压缩到 (0,1)，常用于门控"),
    "tanh": ("y = tanh(x)", "Tanh：压缩到 (-1,1)"),
    "leakyrelu": ("y = max(x, α·x)", "LeakyReLU：负半轴保留小斜率"),
    "prelu": ("y = max(x, 0) + α·min(x, 0)", "PReLU：可学习负半轴斜率"),
    "elu": ("y = x if x>0 else α·(e^x-1)", "ELU：负半轴指数平滑"),
    "selu": ("y = λ·ELU(x)", "SELU：自归一化指数激活"),
    "celu": ("y = max(x,0) + min(0, α·(e^{x/α}-1))", "CELU：连续可导的 ELU 变体"),
    "mish": ("y = x·tanh(softplus(x))", "Mish：平滑非单调激活"),
    "hardswish": ("y = x·clamp(x+3, 0, 6)/6", "Hardswish：MobileNetV3 轻量激活"),
    "hardsigmoid": ("y = clamp(x/6+0.5, 0, 1)", "Hardsigmoid：分段线性门控"),
    "hardtanh": ("y = clamp(x, -1, 1)", "Hardtanh：区间截断"),
    "glu": ("y = a ⊙ σ(b)  (split last dim)", "GLU：两路切分后门控相乘"),
    "softplus": ("y = ln(1+e^x)", "Softplus：平滑 ReLU"),
    "softsign": ("y = x/(1+|x|)", "Softsign：平滑有界激活"),
    "logsigmoid": ("y = ln(σ(x))", "LogSigmoid：对数 Sigmoid"),
    "rrelu": ("y = x if x>0 else α·x", "RReLU：训练期随机斜率"),
    "threshold": ("y = x if x>θ else v", "Threshold：阈值截断"),
}


def _activation_formula(node: dict[str, Any]) -> dict[str, Any]:
    op = _op_short(node).lower()
    formula, semantic = _ACTIVATION_FORMULAS.get(
        op, ("y = f(x)", f"{_op_short(node) or '激活'} 逐元素非线性")
    )
    return {
        "formula": formula,
        "semantic": semantic,
        "axes": "逐元素（任意轴）",
        "tags": ["activation", op or "activation"],
    }


def _softmax_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    dim = _as_int(_param(params, "dim"))
    formula = "y[..., i] = exp(x[..., i]) / Σ_j exp(x[..., j])"
    return {
        "formula": formula + f"  (dim={dim if dim is not None else -1})",
        "semantic": "沿指定轴归一化为概率分布",
        "axes": f"dim={dim if dim is not None else -1}",
        "tags": ["softmax", "normalization"],
    }


def _pool_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    op = _op_short(node).lower()
    ptype = str(_param(params, "type") or "").lower()
    axes = _axes_for_rank(_rank(node)) or "NCHW"
    if _is_global_pool(_noop_ctx(node), node):
        is_max = _is_global_max_pool(_noop_ctx(node), node)
        if op in ("mean", "sum"):
            dim = _as_int(_param(params, "dim"))
            formula = f"y = {op}(x, dim={dim if dim is not None else 1})"
            semantic = "沿序列/空间维平均池化（全局池化）"
        else:
            reducer = "max" if is_max else "mean"
            formula = f"y[n,c] = {reducer}_{{h,w}} x[n,c,h,w]"
            semantic = "全局最大池化：空间维取最大值" if is_max else "全局平均池化：空间维压成 1×1"
        return {
            "formula": formula,
            "semantic": semantic,
            "axes": axes + " → NC",
            "tags": ["pool", "global"],
        }
    kernel = _kernel_text(params)
    stride = _as_int_list(_param(params, "stride", "strides"))
    if op in ("mean", "sum"):
        dim = _as_int(_param(params, "dim"))
        formula = f"y = {op}(x, dim={dim if dim is not None else 1})"
        semantic = "沿指定轴归约池化"
    elif "max" in op or ptype == "max":
        formula = "y[n,c,oh,ow] = max_{kh,kw} x[n,c,oh·s+kh-p, ow·s+kw-p]"
        semantic = f"局部最大池化（k={kernel}）下采样"
    else:
        formula = "y[n,c,oh,ow] = mean_{kh,kw} x[n,c,oh·s+kh-p, ow·s+kw-p]"
        semantic = f"局部平均池化（k={kernel}）下采样"
    if ptype == "adaptive" or "adaptive" in op:
        formula = "y[n,c,oh,ow] = reduce(x[n,c,:,:])  (自适应窗口 → oh×ow)"
        semantic = "自适应池化：目标输出尺寸固定"
    return {
        "formula": formula + f"  (k={kernel}, s={_int_list_text(stride)})",
        "semantic": semantic,
        "axes": axes,
        "tags": ["pool", "max" if ("max" in op or ptype == "max") else "avg"],
    }


def _attention_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    op = _op_short(node).lower()
    heads = _as_int(_param(params, "numHeads"))
    head_dim = _as_int(_param(params, "headDim"))
    if "masked" in op:
        return {
            "formula": "y = masked_fill(scores, mask, -inf)",
            "semantic": "注意力掩码：屏蔽填充位或未来位置",
            "axes": "BHLL",
            "tags": ["attention", "causal_mask"],
        }
    if "matmul" in op or "bmm" in op:
        shape = _first_shape(node) or []
        if len(shape) >= 2 and shape[-1] == shape[-2]:
            formula = "s[b,h,i,j] = Σ_d Q[b,h,i,d]·K[b,h,j,d]"
            semantic = "注意力打分：Q·Kᵀ 缩放点积"
        else:
            formula = "y[b,h,i,d] = Σ_j P[b,h,i,j]·V[b,h,j,d]"
            semantic = "注意力加权求和：P·V"
        return {
            "formula": formula,
            "semantic": semantic,
            "axes": "BHLL / BHLD",
            "tags": ["attention", "matmul"],
        }
    if "multihead" in op or heads or "attention" in op:
        formula = "y = W_o·cat_{h}(softmax(Q_h·K_hᵀ/√d_k)·V_h)"
        dims = []
        if heads:
            dims.append(f"h={heads}")
        if head_dim:
            dims.append(f"d_k={head_dim}")
        return {
            "formula": formula + (f"  ({', '.join(dims)})" if dims else ""),
            "semantic": "多头注意力：Q/K/V 线性投影后按头缩放点积并拼接",
            "axes": "NLC (B,L,E)",
            "tags": ["attention", "multi_head"],
        }
    return {
        "formula": "y = Attn(Q, K, V)",
        "semantic": "注意力算子（打分 + 加权聚合）",
        "axes": "NLC",
        "tags": ["attention"],
    }


def _embedding_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    vocab = _int_param(params, "numEmbeddings", "num_embeddings")
    dim = _int_param(params, "embeddingDim", "embedding_dim")
    return {
        "formula": f"y[..., d] = E[token, d]  (vocab={vocab or '?'}, dim={dim or '?'})",
        "semantic": "词表查表获取嵌入向量",
        "axes": "NLC (B,L,E)",
        "tags": ["embedding", "token"],
    }


def _add_formula(node: dict[str, Any]) -> dict[str, Any]:
    sources = node.get("inShapes")
    count = len(_shapes(sources)) if sources is not None else 2
    if count <= 2:
        formula = "y = x_1 + x_2"
    else:
        formula = "y = Σ_i x_i"
    tags = ["merge", "add"]
    if "residual" in [str(tag) for tag in (node.get("tags") or [])]:
        tags.append("residual")
    return {
        "formula": formula,
        "semantic": "逐元素相加（残差/分支融合）",
        "axes": _axes_for_rank(_rank(node)),
        "tags": tags,
    }


def _concat_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    dim = _as_int(_param(params, "dim"))
    shapes = _shapes(node.get("inShapes"))
    count = len(shapes) if shapes else 2
    parts = ", ".join(f"x_{index + 1}" for index in range(count))
    return {
        "formula": f"y = cat({parts}, dim={dim if dim is not None else 1})",
        "semantic": "沿指定轴拼接多路特征",
        "axes": _axes_for_rank(_rank(node)),
        "tags": ["merge", "concat"],
    }


def _dropout_formula(node: dict[str, Any]) -> dict[str, Any]:
    p = _param(_params(node), "p")
    return {
        "formula": f"y = x ⊙ mask/(1-p),  p={p if p is not None else '?'}",
        "semantic": "训练期随机失活（推理期恒等）",
        "axes": None,
        "tags": ["dropout", "regularization"],
    }


def _flatten_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    start = _as_int(_param(params, "startDim", "start_dim"))
    return {
        "formula": "y = reshape(x, (N, -1))" + (f"  (start_dim={start})" if start is not None else ""),
        "semantic": "展平空间/通道维为特征向量",
        "axes": "N×… → N×F",
        "tags": ["flatten", "reshape"],
    }


def _reshape_formula(node: dict[str, Any]) -> dict[str, Any]:
    op = _op_short(node).lower()
    params = _params(node)
    dim = _as_int(_param(params, "dim"))
    if op in ("transpose", "permute"):
        formula = "y = permute(x, axes)"
        semantic = "转置/换轴，调整张量内存布局"
    elif op in ("chunk", "split"):
        formula = f"y_1..y_k = split(x, k, dim={dim if dim is not None else -1})"
        semantic = "沿指定轴切分为多路张量"
    elif op in ("view", "reshape"):
        formula = "y = reshape(x, shape)"
        semantic = "改变张量形状，不改变元素顺序"
    elif op == "unflatten":
        formula = "y = unflatten(x, dim, sizes)"
        semantic = "把一维拆回多维形状"
    elif op in ("squeeze", "unsqueeze"):
        formula = f"y = {op}(x, dim={dim if dim is not None else 1})"
        semantic = "增删大小为 1 的维度"
    elif op in ("expand", "repeat"):
        formula = "y = broadcast/repeat(x)"
        semantic = "广播或重复扩展张量"
    else:
        formula = "y = reshape(x)"
        semantic = "张量形状变换"
    return {
        "formula": formula,
        "semantic": semantic,
        "axes": "任意",
        "tags": ["reshape", op or "reshape"],
    }


def _rnn_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    op = _op_short(node).lower()
    hidden = _int_param(params, "hiddenFeatures", "hidden_size")
    layers = _int_param(params, "numLayers", "num_layers")
    suffix = []
    if hidden:
        suffix.append(f"hidden={hidden}")
    if layers:
        suffix.append(f"layers={layers}")
    note = f"  ({', '.join(suffix)})" if suffix else ""
    if "lstm" in op:
        formula = "(h_t, c_t) = LSTM(x_t, (h_{t-1}, c_{t-1}))"
        semantic = "LSTM：遗忘/输入/输出门更新细胞状态"
    elif "gru" in op:
        formula = "h_t = GRU(x_t, h_{t-1})"
        semantic = "GRU：更新门/重置门更新隐状态"
    else:
        formula = "h_t = tanh(W·x_t + U·h_{t-1})"
        semantic = "RNN：按时间步更新隐藏状态"
    return {
        "formula": formula + note,
        "semantic": semantic,
        "axes": "NLC (B,T,E)",
        "tags": ["rnn", "sequence", op or "rnn"],
    }


def _upsample_formula(node: dict[str, Any]) -> dict[str, Any]:
    params = _params(node)
    scale = _param(params, "scaleFactor")
    mode = _param(params, "mode")
    note = []
    if scale is not None:
        note.append(f"scale={scale}")
    if mode:
        note.append(f"mode={mode}")
    return {
        "formula": "y = resize(x, scale/size)" + (f"  ({', '.join(note)})" if note else ""),
        "semantic": "上采样：放大空间/序列分辨率",
        "axes": _axes_for_rank(_rank(node)),
        "tags": ["upsample"],
    }


def _positional_formula(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula": "y = x + PE(pos, d)",
        "semantic": "位置编码：向 token 表示注入顺序信息",
        "axes": "NLC",
        "tags": ["positional", "sequence"],
    }


def _block_formula(node: dict[str, Any]) -> dict[str, Any]:
    op = _op_short(node) or "Block"
    params = _params(node)
    embed = _int_param(params, "embedDim", "d_model")
    heads = _int_param(params, "numHeads", "nhead")
    note = []
    if embed:
        note.append(f"d={embed}")
    if heads:
        note.append(f"h={heads}")
    return {
        "formula": f"y = {op}(x)" + (f"  ({', '.join(note)})" if note else ""),
        "semantic": "融合的复合层（Transformer/自定义 Block，内部已折叠）",
        "axes": _axes_for_rank(_rank(node), sequence=True),
        "tags": ["block", "composite"],
    }


def _generic_formula(node: dict[str, Any]) -> dict[str, Any]:
    kind = _node_kind(node)
    op = _op_short(node) or kind
    if kind == "custom":
        name = _norm_text(node.get("name"))
        if "roll" in name:
            formula = "y = roll(x, shifts)"
            semantic = "循环移位：窗口/序列平移"
            tag = "roll"
        elif _is_mul(node):
            formula = "y = x_1 ⊙ x_2"
            semantic = "逐元素相乘（门控/缩放）"
            tag = "mul"
        else:
            formula = f"y = {op}(x)"
            semantic = "自定义算子：图中未展开其内部实现"
            tag = "custom"
    else:
        formula = f"y = {op}(x)"
        semantic = f"{op} 算子"
        tag = kind
    return {
        "formula": formula,
        "semantic": semantic,
        "axes": _axes_for_rank(_rank(node)),
        "tags": [kind, tag],
    }


_FORMULA_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "conv": _conv_formula,
    "deconv": _deconv_formula,
    "linear": _linear_formula,
    "norm": _norm_formula,
    "activation": _activation_formula,
    "softmax": _softmax_formula,
    "pool": _pool_formula,
    "attention": _attention_formula,
    "embedding": _embedding_formula,
    "add": _add_formula,
    "concat": _concat_formula,
    "dropout": _dropout_formula,
    "flatten": _flatten_formula,
    "reshape": _reshape_formula,
    "rnn": _rnn_formula,
    "upsample": _upsample_formula,
    "positional": _positional_formula,
    "block": _block_formula,
}

_FORMULA_HANDLERS["ffn"] = lambda node: {
    "formula": "y = W2·Act(W1·x)",
    "semantic": "位置前馈网络：先扩张后收缩",
    "axes": "NLC (B,L,E)",
    "tags": ["ffn", "expansion"],
}
_FORMULA_HANDLERS["mlp"] = lambda node: {
    "formula": "y = W_n·Act(⋯Act(W_1·x))",
    "semantic": "多层感知机：逐层线性 + 激活",
    "axes": "N×F",
    "tags": ["mlp", "linear"],
}


def node_formula(node: dict) -> dict:
    """Return ``{"formula", "semantic", "axes", "tags"}`` for a single node.

    The formula is a compact ASCII/Unicode expression with axis and kernel
    annotations; ``semantic`` is a short Chinese phrase (never empty boilerplate).
    """
    if not isinstance(node, dict):
        return {
            "formula": "y = ?",
            "semantic": "未知节点",
            "axes": None,
            "tags": ["unknown"],
        }
    kind = _node_kind(node)
    try:
        if kind == "input":
            shape = _first_shape(node) or _first_shape(node, "inShapes")
            result = {
                "formula": f"x = Input({', '.join(str(dim) for dim in shape or ['?'])})",
                "semantic": "网络输入张量",
                "axes": _axes_for_rank(len(shape) if shape else None),
                "tags": ["boundary", "input"],
            }
        elif kind == "output":
            shape = _first_shape(node) or []
            result = {
                "formula": f"y = Output({_shape_text(shape)})",
                "semantic": "网络输出张量",
                "axes": _axes_for_rank(len(shape) if shape else None),
                "tags": ["boundary", "output"],
            }
        elif kind == "container":
            result = {
                "formula": "y = x",
                "semantic": "容器层：仅组织子模块，本身无计算",
                "axes": None,
                "tags": ["container"],
            }
        elif kind == "unknown":
            result = {
                "formula": "y = ?",
                "semantic": "未知算子（图中无更多语义信息）",
                "axes": None,
                "tags": ["unknown"],
            }
        else:
            handler = _FORMULA_HANDLERS.get(kind)
            result = handler(node) if handler is not None else _generic_formula(node)
    except Exception:  # pragma: no cover - malformed node fields must never crash
        result = _generic_formula(node)
    formula = str(result.get("formula") or "?")
    semantic = str(result.get("semantic") or "未知算子")
    axes = result.get("axes")
    axes = str(axes) if isinstance(axes, str) and axes else None
    tags: list[str] = []
    for tag in result.get("tags") or []:
        if isinstance(tag, str) and tag and tag not in tags:
            tags.append(tag)
    if not tags:
        tags = [kind]
    return {"formula": formula, "semantic": semantic, "axes": axes, "tags": tags[:8]}


# --------------------------------------------------------------------------- #
# graph analysis / public API
# --------------------------------------------------------------------------- #
def _analyze_graph(graph: Any, kb: dict[str, Any]) -> dict[str, Any]:
    ctx = _Ctx(graph)
    patterns = _finalize_patterns(ctx, _detect_patterns(ctx))
    facts, details = _structure_facts(ctx, patterns)
    families, family_status = _build_candidates(ctx, facts, details, patterns, kb)
    group_patterns = _group_patterns(ctx, patterns)
    pseudocode = _pseudocode(ctx, patterns, group_patterns)
    valid_family = family_status != "unknown" and bool(families)
    if valid_family:
        top = families[0]
        family_id = str(top.get("id") or "unknown")
        family_label = str(top.get("label") or _FAMILY_LABELS.get(family_id, "Unknown"))
        confidence = float(top.get("confidence") or 0.0)
    else:
        family_id = "unknown"
        family_label = "Unknown"
        confidence = 0.0
    tags = _algorithm_tags(ctx, families if valid_family else [], patterns)
    reason = _unknown_reason(ctx, families, patterns)
    return {
        "ctx": ctx,
        "family": family_id,
        "familyLabel": family_label,
        "confidence": confidence,
        "families": families,
        "tags": tags,
        "patterns": patterns,
        "group_patterns": group_patterns,
        "pseudocode": pseudocode,
        "unknown_reason": reason,
    }


def classify_families(graph: dict) -> list[dict]:
    """Rank algorithm families for ``graph`` (top 3 ``{id,label,confidence,evidence}``).

    Scoring combines: name/alias match, knowledge-base ``required_patterns`` /
    ``optional_patterns`` (mapped to structural signals or ``required_ops``),
    and the graph's operator signature.  Unrecognised graphs return only the
    weak real candidates (or an empty list); ``unknown`` is never returned as a
    family candidate — ``explain_graph`` reports it via ``family`` /
    ``unknown_reason`` with confidence ``0.0``.
    """
    analysis = _analyze_graph(graph if isinstance(graph, dict) else {}, _knowledge())
    return analysis["families"]


def enrich_algorithms(graph: dict, *, force: bool = False) -> dict:
    """Write family, patterns, pseudocode and node formulas into ``*.meta``.

    ``nodes``/``edges``/``params``/``metrics`` are never modified.  When
    ``model.meta.family`` already exists and ``force`` is false the graph is
    returned untouched (idempotent for repeated CLI/report runs).
    """
    if not isinstance(graph, dict):
        return graph
    model = graph.get("model")
    existing_meta = model.get("meta") if isinstance(model, dict) and isinstance(model.get("meta"), dict) else None
    if (
        existing_meta is not None
        and not force
        and existing_meta.get("family")
        and existing_meta.get("family") != "unknown"
    ):
        return graph

    analysis = _analyze_graph(graph, _knowledge())

    if isinstance(model, dict):
        meta = model.get("meta") if isinstance(model.get("meta"), dict) else None
        if meta is None:
            meta = {}
            model["meta"] = meta
        meta["family"] = analysis["family"]
        meta["familyLabel"] = analysis["familyLabel"]
        meta["familyConfidence"] = round(float(analysis["confidence"]), 2)
        meta["confidence"] = meta["familyConfidence"]
        meta["algorithmTags"] = list(analysis["tags"])
        meta["families"] = analysis["families"]
        meta["patterns"] = analysis["patterns"]
        meta["pseudocode"] = analysis["pseudocode"]
        if analysis["unknown_reason"]:
            meta["unknownReason"] = analysis["unknown_reason"]
        elif "unknownReason" in meta:
            del meta["unknownReason"]
        if analysis["family"] != "unknown" and analysis["families"]:
            top = analysis["families"][0]
            if top.get("evidence"):
                meta["familyEvidence"] = list(top.get("evidence") or [])
        elif "familyEvidence" in meta:
            del meta["familyEvidence"]

    raw_nodes = graph.get("nodes")
    if isinstance(raw_nodes, list):
        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            info = node_formula(node)
            node_meta = node.get("meta") if isinstance(node.get("meta"), dict) else None
            if node_meta is None:
                node_meta = {}
                node["meta"] = node_meta
            node_meta["formula"] = info["formula"]
            node_meta["semantic"] = info["semantic"]
            node_meta["axes"] = info["axes"]
            node_meta["tags"] = info["tags"]

    raw_groups = graph.get("groups")
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if not isinstance(group, dict):
                continue
            group_id = group.get("id")
            pattern = analysis["group_patterns"].get(group_id) if isinstance(group_id, str) else None
            if not pattern:
                continue
            group_meta = group.get("meta") if isinstance(group.get("meta"), dict) else None
            if group_meta is None:
                group_meta = {}
                group["meta"] = group_meta
            group_meta["pattern"] = pattern

    return graph


def explain_graph(graph: dict) -> dict:
    """Return a pure-JSON algorithm summary for CLI/report/web consumption."""
    graph = graph if isinstance(graph, dict) else {}
    analysis = _analyze_graph(graph, _knowledge())
    model = graph.get("model")
    meta = model.get("meta") if isinstance(model, dict) and isinstance(model.get("meta"), dict) else {}

    family = meta.get("family") if isinstance(meta.get("family"), str) and meta.get("family") else analysis["family"]
    label = (
        meta.get("familyLabel")
        if isinstance(meta.get("familyLabel"), str) and meta.get("familyLabel")
        else analysis["familyLabel"]
    )
    confidence = meta.get("familyConfidence") if isinstance(meta.get("familyConfidence"), (int, float)) else analysis["confidence"]
    stored_families = meta.get("families") if isinstance(meta.get("families"), list) else []
    families = [
        candidate
        for candidate in (stored_families or analysis["families"])
        if isinstance(candidate, dict) and candidate.get("id") != "unknown"
    ]
    unknown_reason = analysis["unknown_reason"]
    if str(family) == "unknown":
        # ``unknown`` is a status, not a high-confidence family prediction.
        confidence = 0.0
        label = "Unknown"
        if not unknown_reason:
            unknown_reason = "未识别出有效家族：结构证据不足"
    tags = meta.get("algorithmTags") if isinstance(meta.get("algorithmTags"), list) and meta.get("algorithmTags") else analysis["tags"]
    patterns = meta.get("patterns") if isinstance(meta.get("patterns"), list) and meta.get("patterns") else analysis["patterns"]
    pseudocode = (
        meta.get("pseudocode")
        if isinstance(meta.get("pseudocode"), list) and meta.get("pseudocode")
        else analysis["pseudocode"]
    )

    groups = []
    for group in analysis["ctx"].groups:
        group_id = group.get("id")
        if not isinstance(group_id, str):
            continue
        stored = group.get("meta") if isinstance(group.get("meta"), dict) else {}
        pattern = stored.get("pattern")
        if not isinstance(pattern, dict):
            pattern = analysis["group_patterns"].get(group_id)
        groups.append(
            {
                "id": group_id,
                "name": group.get("name"),
                "label": group.get("label"),
                "kind": group.get("kind"),
                "depth": group.get("depth"),
                "nodeCount": len(analysis["ctx"].subtree_nodes(group_id)),
                "pattern": pattern,
            }
        )

    return {
        "family": str(family),
        "familyLabel": str(label),
        "familyConfidence": round(float(confidence or 0.0), 2),
        "confidence": round(float(confidence or 0.0), 2),
        "families": families,
        "tags": [str(tag) for tag in tags],
        "patterns": patterns,
        "pseudocode": pseudocode,
        "groups": groups,
        "unknown_reason": unknown_reason,
    }


def render_explain_md(graph: dict) -> str:
    """Render a compact Markdown algorithm section (families, patterns, pseudocode)."""
    data = explain_graph(graph)
    lines: list[str] = ["### 算法理解", ""]
    families = data.get("families") or []
    if families and data.get("family") != "unknown":
        lines.append("**家族候选**")
        lines.append("")
        lines.append("| 家族 | id | 置信度 | 结构证据 |")
        lines.append("| --- | --- | --- | --- |")
        for family in families[:3]:
            evidence = "；".join(str(item) for item in (family.get("evidence") or [])[:2])
            evidence = evidence.replace("|", "/") or "—"
            lines.append(
                f"| {family.get('label') or family.get('id')} | `{family.get('id')}` | "
                f"{float(family.get('confidence') or 0):.2f} | {evidence} |"
            )
        lines.append("")
    else:
        lines.append(f"- **家族**: {data.get('familyLabel')} (`{data.get('family')}`)")
        lines.append("")
    tags = data.get("tags") or []
    if tags:
        lines.append("- **标签**: " + ", ".join(str(tag) for tag in tags[:10]))
        lines.append("")
    if data.get("unknown_reason"):
        lines.append(f"> {data['unknown_reason']}")
        lines.append("")

    patterns = [pattern for pattern in (data.get("patterns") or []) if isinstance(pattern, dict)]
    aggregated: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        kind = str(pattern.get("kind") or "")
        entry = aggregated.setdefault(
            kind,
            {
                "kind": kind,
                "label": str(pattern.get("label") or kind),
                "formula": str(pattern.get("formula") or "?"),
                "confidence": 0.0,
                "count": 0,
            },
        )
        entry["count"] += 1
        entry["confidence"] = max(
            float(entry["confidence"]), float(pattern.get("confidence") or 0)
        )
    key_patterns = sorted(
        aggregated.values(),
        key=lambda entry: (-float(entry["confidence"]), entry["kind"]),
    )[:10]
    if key_patterns:
        lines.append("**关键组件**")
        lines.append("")
        lines.append("| 组件 | 类型 | 公式 | 置信度 |")
        lines.append("| --- | --- | --- | --- |")
        for pattern in key_patterns:
            label = str(pattern.get("label") or pattern.get("kind") or "").replace("|", "/")
            count = int(pattern.get("count") or 0)
            if count > 1:
                label = f"{label} ×{count}"
            kind = str(pattern.get("kind") or "")
            formula = str(pattern.get("formula") or "?").replace("|", "/")
            lines.append(
                f"| {label} | `{kind}` | `{formula}` | {float(pattern.get('confidence') or 0):.2f} |"
            )
        lines.append("")

    pseudocode = data.get("pseudocode") or []
    if pseudocode:
        lines.append("**Forward 伪代码**")
        lines.append("")
        lines.append("```python")
        lines.extend(str(line) for line in pseudocode)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ALGORITHM_VERSION",
    "CANONICAL_FAMILIES",
    "PATTERN_SPECS",
    "classify_families",
    "enrich_algorithms",
    "explain_graph",
    "node_formula",
    "render_explain_md",
]
