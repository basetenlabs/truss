"""
Functional transformer_engine stub for environments where the real TE
binary wheels are incompatible with the installed PyTorch/CUDA versions.

Provides:
  - te.pytorch.Linear            (nn.Module with weight/bias)
  - te.pytorch.LayerNormLinear   (nn.Module with layer_norm_weight + weight/bias)
  - te.pytorch.RMSNorm / LayerNorm  (real PyTorch norm modules)
  - is_te_min_version()          (version comparison helper)
  - get_cpu_offload_context()    (returns a no-op context + sync fn)
  - All other sub-module attributes resolve to _MagicModule / _DummyClass
    so that the many conditional TE imports in Megatron-Core don't raise.
"""
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


__version__ = "2.2.0"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _parse_ver(v: str):
    """Return a tuple of ints from a version string like '1.10.0.dev0'."""
    clean = v.split("+")[0]           # strip "+gitXXX"
    parts = clean.replace(".dev", ".").split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    return tuple(nums)


_STUB_VERSION = _parse_ver(__version__)


def is_te_min_version(min_version: str, check_equality: bool = True) -> bool:
    """Return True if the stub version >= *min_version*."""
    req = _parse_ver(min_version)
    if check_equality:
        return _STUB_VERSION >= req
    return _STUB_VERSION > req


def get_te_version():
    """Return the stub version as a packaging.version.Version-like object."""
    try:
        from packaging.version import Version
        return Version(__version__)
    except Exception:
        return _STUB_VERSION


# ---------------------------------------------------------------------------
# Functional stub classes that act as real nn.Modules
# ---------------------------------------------------------------------------

class _TELinear(nn.Module):
    """Stub for ``te.pytorch.Linear``.

    Accepts every kwarg the real TE class does; ignores TE-specific ones but
    honours ``in_features``, ``out_features``, ``bias``, ``init_method``, and
    ``params_dtype`` to create proper weight/bias parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_bias: bool = False,
        init_method=None,
        params_dtype=None,
        # Ignored TE-specific kwargs
        sequence_parallel=False,
        fuse_wgrad_accumulation=False,
        tp_group=None,
        tp_size: int = 1,
        get_rng_state_tracker=None,
        parallel_mode=None,
        rng_tracker_name=None,
        eps=None,
        zero_centered_gamma: bool = False,
        return_layernorm_output: bool = False,
        delay_wgrad_compute: bool = False,
        normalization=None,
        **ignored,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.return_bias = return_bias
        self.tp_size = tp_size

        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        if init_method is not None:
            try:
                init_method(self.weight)
            except Exception:
                nn.init.kaiming_uniform_(self.weight, a=0.01)
        else:
            nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x, is_first_microbatch=None):
        # Return the raw tensor.  Megatron's TELinear.forward() calls
        # super().forward() and then wraps the result as (out, None) itself —
        # so we must NOT return a tuple here to avoid a nested-tuple bug.
        return F.linear(x, self.weight, self.bias)

    def backward_dw(self):
        pass


class _TELayerNormLinear(_TELinear):
    """Stub for ``te.pytorch.LayerNormLinear``.

    Extends ``_TELinear`` with a fused ``layer_norm_weight`` (and optionally
    ``layer_norm_bias``) parameter, mirroring the TE API that Megatron-Core's
    ``TELayerNormColumnParallelLinear`` relies on.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalization=None,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        params_dtype=None,
        return_layernorm_output: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            normalization=normalization,
            params_dtype=params_dtype,
            **kwargs,
        )
        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self._norm_type = normalization or "LayerNorm"
        self._eps = eps

        self.layer_norm_weight = nn.Parameter(torch.ones(in_features, dtype=dtype))
        if self._norm_type == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.zeros(in_features, dtype=dtype))
        else:
            self.register_parameter("layer_norm_bias", None)

    def forward(self, x, is_first_microbatch=None):
        # Apply fused layer norm then linear.
        # Return the raw tensor (not a tuple) — megatron's wrapper adds the
        # (out, None) tuple in its own forward() after calling super().
        lnw = self.layer_norm_weight
        lnb = self.layer_norm_bias
        if self._norm_type == "RMSNorm":
            normed = F.rms_norm(x, (x.shape[-1],), lnw, self._eps)
        else:
            normed = F.layer_norm(x, (x.shape[-1],), lnw, lnb, self._eps)
        return F.linear(normed, self.weight, self.bias)


class _TERMSNorm(nn.Module):
    """Stub for ``te.pytorch.RMSNorm`` — delegates to ``torch.nn.RMSNorm``."""

    def __init__(self, hidden_size, eps: float = 1e-5, zero_centered_gamma: bool = False,
                 sequence_parallel: bool = False, params_dtype=None, **ignored):
        super().__init__()
        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self._norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)

    def forward(self, x):
        return self._norm(x)

    @property
    def weight(self):
        return self._norm.weight

    @property
    def bias(self):
        return None


class _TELayerNorm(nn.Module):
    """Stub for ``te.pytorch.LayerNorm`` — delegates to ``torch.nn.LayerNorm``."""

    def __init__(self, hidden_size, eps: float = 1e-5, zero_centered_gamma: bool = False,
                 sequence_parallel: bool = False, params_dtype=None, **ignored):
        super().__init__()
        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self._norm = nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)

    def forward(self, x):
        return self._norm(x)

    @property
    def weight(self):
        return self._norm.weight

    @property
    def bias(self):
        return self._norm.bias


# ---------------------------------------------------------------------------
# DotProductAttention — real attention computation used by TEDotProductAttention
# ---------------------------------------------------------------------------

class _TEDotProductAttention(nn.Module):
    """Stub for ``te.pytorch.DotProductAttention``.

    Provides real flash/SDPA-backed attention so that megatron's
    ``TEDotProductAttention`` (which inherits from this class) produces
    actual tensors rather than _DummyObj instances.

    Supports both:
      - ``sbhd``/``bshd`` format  (standard non-packed batches)
      - ``thd`` format            (packed sequences, padding_free=True)
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float = 0.0,
        attn_mask_type: str = "causal",
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker=None,
        tp_group=None,
        layer_number: int = 1,
        num_gqa_groups: int = None,
        attention_type: str = "self",
        softmax_scale: float = None,
        **ignored,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_gqa_groups = num_gqa_groups if num_gqa_groups is not None else num_attention_heads
        # kv_channels is head_dim when an int, or (k_dim, v_dim) tuple from TE>=1.10
        if isinstance(kv_channels, (tuple, list)):
            self.head_dim = kv_channels[0]
        else:
            self.head_dim = kv_channels
        self.attention_dropout = attention_dropout
        self._default_attn_mask_type = attn_mask_type
        self.softmax_scale = softmax_scale or (self.head_dim ** -0.5)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type: str = None,
        attention_bias=None,
        # packed-sequence kwargs (from PackedSeqParams fields)
        qkv_format: str = None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **ignored,
    ):
        """Compute scaled dot-product attention.

        Handles both packed (thd) and unpacked (sbhd/bshd) formats.
        Returns the same dtype as *query*.
        """
        mask_type = attn_mask_type or self._default_attn_mask_type
        is_causal = "causal" in mask_type.lower() if isinstance(mask_type, str) else True
        dropout_p = self.attention_dropout if self.training else 0.0

        # Prefer padded cu_seqlens if available
        _cu_q = cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q
        _cu_kv = cu_seqlens_kv_padded if cu_seqlens_kv_padded is not None else cu_seqlens_kv

        fmt = qkv_format or "sbhd"

        if fmt == "thd" and _cu_q is not None:
            return self._forward_thd(query, key, value, _cu_q, _cu_kv, is_causal, dropout_p)
        elif fmt in ("bshd",):
            return self._forward_bshd(query, key, value, is_causal, dropout_p)
        else:
            # Default: sbhd  [s, b, np, hn]
            return self._forward_sbhd(query, key, value, is_causal, dropout_p)

    def _expand_gqa(self, k, v, dim):
        """Repeat k/v heads to match query head count along *dim*."""
        ratio = self.num_heads // self.num_gqa_groups
        if ratio > 1:
            k = k.repeat_interleave(ratio, dim=dim)
            v = v.repeat_interleave(ratio, dim=dim)
        return k, v

    def _forward_sbhd(self, query, key, value, is_causal, dropout_p):
        """sbhd format: [s, b, np, hn] -> [s, b, np*hn]."""
        k, v = self._expand_gqa(key, value, dim=2)
        sq, b, np, hn = query.shape
        # [s, b, np, hn] -> [b, np, s, hn]
        q_t = query.permute(1, 2, 0, 3).contiguous()
        k_t = k.permute(1, 2, 0, 3).contiguous()
        v_t = v.permute(1, 2, 0, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
        )
        # [b, np, sq, hn] -> [sq, b, np*hn]
        return out.permute(2, 0, 1, 3).reshape(sq, b, np * hn)

    def _forward_bshd(self, query, key, value, is_causal, dropout_p):
        """bshd format: [b, s, np, hn] -> [b, s, np*hn]."""
        k, v = self._expand_gqa(key, value, dim=2)
        b, sq, np, hn = query.shape
        # [b, s, np, hn] -> [b, np, s, hn]
        q_t = query.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
        )
        # [b, np, s, hn] -> [b, s, np*hn]
        return out.permute(0, 2, 1, 3).reshape(b, sq, np * hn)

    def _forward_thd(self, query, key, value, cu_seqlens_q, cu_seqlens_kv, is_causal, dropout_p):
        """thd format: [t, np, hn] -> [t, np, hn] (variable-length sequences)."""
        k, v = self._expand_gqa(key, value, dim=1)
        batch_size = cu_seqlens_q.numel() - 1
        outputs = []
        for i in range(batch_size):
            qs = cu_seqlens_q[i].item()
            qe = cu_seqlens_q[i + 1].item()
            ks = cu_seqlens_kv[i].item()
            ke = cu_seqlens_kv[i + 1].item()
            # [sq, np, hn] -> [1, np, sq, hn]
            q_i = query[qs:qe].unsqueeze(0).transpose(1, 2)
            k_i = k[ks:ke].unsqueeze(0).transpose(1, 2)
            v_i = v[ks:ke].unsqueeze(0).transpose(1, 2)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=self.softmax_scale,
            )
            # [1, np, sq, hn] -> [sq, np, hn]
            outputs.append(out_i.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)  # [t, np, hn]


# ---------------------------------------------------------------------------
# Non-module stubs
# ---------------------------------------------------------------------------

class _DummyClass:
    """Generic no-op class — stand-in for TE tensor types, etc."""
    def __init__(self, *a, **kw):
        pass
    def __call__(self, *a, **kw):
        return self
    @classmethod
    def __class_getitem__(cls, item):
        return cls


def _dummy_fn(*a, **kw):
    return None


class _DummyObj:
    """Returned when a generic stub callable is invoked."""
    def __iter__(self):
        return iter([])
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __call__(self, *a, **kw):
        return self


class _MagicModule(types.ModuleType):
    """
    A module that never raises AttributeError.

    CamelCase attr  → fresh _DummyClass subclass  (usable as base class)
    lowercase attr  → child _MagicModule           (so te.pytorch.ops.X works)

    Instances are also callable (return a _DummyObj) so that functions imported
    from this stub (e.g. get_cpu_offload_context) work when called.
    """
    def __call__(self, *args, **kwargs):
        return _DummyObj()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].isupper():
            val = type(name, (_DummyClass,), {})
        else:
            full = f"{self.__name__}.{name}"
            val = _MagicModule(full)
            sys.modules[full] = val
        object.__setattr__(self, name, val)
        return val


def _stub(name: str) -> _MagicModule:
    """Register a _MagicModule at *name* and wire it to its parent."""
    mod = _MagicModule(name)
    sys.modules[name] = mod
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = sys.modules.get(parts[0])
        if parent is not None:
            try:
                object.__setattr__(parent, parts[1], mod)
            except (AttributeError, TypeError):
                pass
    return mod


# ---------------------------------------------------------------------------
# Pre-register sub-modules
# ---------------------------------------------------------------------------

_SUBS = [
    "transformer_engine.pytorch",
    "transformer_engine.pytorch.attention",
    "transformer_engine.pytorch.attention.rope",
    "transformer_engine.pytorch.module",
    "transformer_engine.pytorch.module.base",
    "transformer_engine.pytorch.tensor",
    "transformer_engine.pytorch.tensor.float8_tensor",
    "transformer_engine.pytorch.tensor.mxfp8_tensor",
    "transformer_engine.pytorch.tensor.nvfp4_tensor",
    "transformer_engine.pytorch.tensor.utils",
    "transformer_engine.pytorch.float8_tensor",
    "transformer_engine.pytorch.fp8",
    "transformer_engine.pytorch.distributed",
    "transformer_engine.pytorch.cross_entropy",
    "transformer_engine.pytorch.graph",
    "transformer_engine.pytorch.optimizers",
    "transformer_engine.pytorch.permutation",
    "transformer_engine.pytorch.router",
    "transformer_engine.pytorch.cpu_offload",
    "transformer_engine.pytorch.cpp_extensions",
    "transformer_engine.pytorch.ops",
]
for _s in _SUBS:
    _stub(_s)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Utility functions used by Megatron-Core
# ---------------------------------------------------------------------------

class _SplitAlongDimFn(torch.autograd.Function):
    """Stub for te.pytorch.attention._SplitAlongDim — splits along a dim."""
    @staticmethod
    def forward(ctx, tensor, dim, split_sizes):
        ctx.dim = dim
        ctx.split_sizes = list(split_sizes)
        return torch.split(tensor, ctx.split_sizes, dim=dim)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return torch.cat(grad_outputs, dim=ctx.dim), None, None

_SplitAlongDim = _SplitAlongDimFn


# ---------------------------------------------------------------------------
# Wire functional classes into te.pytorch
# ---------------------------------------------------------------------------

_te_pt = sys.modules["transformer_engine.pytorch"]

# Real nn.Module classes (must be set before _MagicModule's __getattr__ fires)
object.__setattr__(_te_pt, "Linear",                _TELinear)
object.__setattr__(_te_pt, "LayerNormLinear",        _TELayerNormLinear)
object.__setattr__(_te_pt, "RMSNorm",                _TERMSNorm)
object.__setattr__(_te_pt, "LayerNorm",              _TELayerNorm)
object.__setattr__(_te_pt, "DotProductAttention",    _TEDotProductAttention)

# Version helpers
object.__setattr__(_te_pt, "is_te_min_version", is_te_min_version)
object.__setattr__(_te_pt, "get_te_version",    get_te_version)

# SplitAlongDim is imported from te.pytorch.attention._SplitAlongDim.apply
_te_attn = sys.modules["transformer_engine.pytorch.attention"]
object.__setattr__(_te_attn, "_SplitAlongDim", _SplitAlongDim)

# apply_rotary_pos_emb: megatron imports this from te.pytorch.attention when
# HAVE_TE=True and is_te_min_version("2.3.0") is False (which is our case: 2.2.0).
# The _MagicModule default would return _DummyObj when called, breaking save_for_backward.
# Delegate to megatron's native rope implementation instead.
def _apply_rotary_pos_emb_stub(t, freqs, *args, **kwargs):
    try:
        from megatron.core.models.common.embeddings.rope_utils import (
            apply_rotary_pos_emb as _native,
        )
        return _native(t, freqs, *args, **kwargs)
    except Exception:
        return t  # fallback: return input tensor unchanged

object.__setattr__(_te_attn, "apply_rotary_pos_emb", _apply_rotary_pos_emb_stub)

# Also wire into the rope sub-module (used when is_te_min_version("2.3.0") is True,
# but set it anyway so any import path works).
_te_attn_rope = sys.modules["transformer_engine.pytorch.attention.rope"]
object.__setattr__(_te_attn_rope, "apply_rotary_pos_emb", _apply_rotary_pos_emb_stub)

# get_cpu_offload_context returns (context_manager, sync_fn) — megatron unpacks as 2-tuple.
def _get_cpu_offload_context_stub(*a, **kw):
    return _DummyObj(), _dummy_fn

object.__setattr__(
    sys.modules["transformer_engine.pytorch.cpu_offload"],
    "get_cpu_offload_context",
    _get_cpu_offload_context_stub,
)

# Megatron imports FusedAdam/FusedSGD from te.pytorch.optimizers as the
# preferred optimizer — wrap torch's AdamW/SGD accepting TE-specific kwargs.
class _FusedAdam(torch.optim.AdamW):
    """Stub for te.pytorch.optimizers.FusedAdam — wraps torch.optim.AdamW."""
    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999),
                 eps=1e-8, adam_w_mode=True, weight_decay=0.01, amsgrad=False,
                 set_grad_none=True, capturable=False, master_weights=False,
                 **ignored):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

class _FusedSGD(torch.optim.SGD):
    """Stub for te.pytorch.optimizers.FusedSGD — wraps torch.optim.SGD."""
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, **ignored):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

_te_opt = sys.modules["transformer_engine.pytorch.optimizers"]
object.__setattr__(_te_opt, "FusedAdam", _FusedAdam)
object.__setattr__(_te_opt, "FusedSGD",  _FusedSGD)

# Megatron's clip_grads.py imports multi_tensor_applier / multi_tensor_l2norm /
# multi_tensor_scale from te.pytorch.optimizers (preferred over apex).  The
# _MagicModule defaults would return _DummyObj which cannot be unpacked as a
# 2-tuple.  Delegate to megatron's local pure-PyTorch implementations instead.
def _multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    try:
        from megatron.core.utils import local_multi_tensor_applier
        return local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args)
    except Exception:
        return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)

def _multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    try:
        from megatron.core.utils import local_multi_tensor_l2_norm
        return local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args)
    except Exception:
        import torch as _torch
        l2 = [[(t.float().norm(2)) for t in tl] for tl in tensor_lists]
        norm = _torch.norm(_torch.tensor(l2))
        return _torch.tensor([float(norm)], dtype=_torch.float, device="cuda"), None

def _multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    try:
        from megatron.core.utils import local_multi_tensor_scale
        return local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)
    except Exception:
        for src, dst in zip(tensor_lists[0], tensor_lists[1]):
            dst.copy_(src * scale)

object.__setattr__(_te_opt, "multi_tensor_applier", _multi_tensor_applier)
object.__setattr__(_te_opt, "multi_tensor_l2norm",  _multi_tensor_l2norm)
object.__setattr__(_te_opt, "multi_tensor_scale",   _multi_tensor_scale)

# Also expose is_te_min_version at package level for:
#   from transformer_engine.pytorch import is_te_min_version  (done above)
# and direct:
#   from transformer_engine import is_te_min_version  (handled by __getattr__ below)


# ---------------------------------------------------------------------------
# Module-level __getattr__ for ``import transformer_engine as te; te.pytorch``
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    if name == "is_te_min_version":
        return is_te_min_version
    if name == "get_te_version":
        return get_te_version
    full = f"transformer_engine.{name}"
    if full in sys.modules:
        return sys.modules[full]
    mod = _stub(full)
    globals()[name] = mod
    return mod
