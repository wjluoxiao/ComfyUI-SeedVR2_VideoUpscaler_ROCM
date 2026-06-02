"""
Compatibility module for SeedVR2
Contains FP8/FP16 compatibility layers and wrappers for different model architectures

Extracted from: seedvr2.py (lines 1045-1630)
"""

# Compatibility shims - Must run before any torch/diffusers import
import sys
import types
import importlib.machinery


def ensure_triton_compat():
    """Create minimal triton.ops stubs only if missing, to allow bitsandbytes import."""
    if 'triton.ops.matmul_perf_model' in sys.modules:
        return
    
    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    
    if 'triton.ops' not in sys.modules:
        sys.modules['triton.ops'] = types.ModuleType('triton.ops')
    
    matmul_perf = types.ModuleType('triton.ops.matmul_perf_model')
    matmul_perf.early_config_prune = lambda configs, *a, **kw: configs
    matmul_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    
    sys.modules['triton.ops'].matmul_perf_model = matmul_perf
    sys.modules['triton.ops.matmul_perf_model'] = matmul_perf


def ensure_bitsandbytes_safe():
    """
    Pre-test bitsandbytes; stub if broken to prevent import conflicts.
    
    On some systems (e.g., ROCm without proper binaries), bitsandbytes registers
    PyTorch kernels during import then fails. If another node already triggered 
    this partial load, re-importing causes kernel registration conflicts.
    
    This shim catches such failures and stubs the module so diffusers can load
    gracefully without bitsandbytes quantization support.
    """
    if 'bitsandbytes' in sys.modules:
        return  # Already loaded or stubbed
    
    try:
        import bitsandbytes
        # Success - bitsandbytes works, other nodes can use it
    except (ImportError, OSError, RuntimeError, ValueError):
        # Installation broken, not present, or version detection failed - create stub
        stub = types.ModuleType('bitsandbytes')
        stub.__spec__ = importlib.machinery.ModuleSpec('bitsandbytes', None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__version__ = "0.0.0"
        sys.modules['bitsandbytes'] = stub


# Run all shims immediately on import, before torch/diffusers
ensure_triton_compat()
ensure_bitsandbytes_safe()


import torch
import os


# SageAttention & Triton Compatibility Layer

# patch: SageAttention1 （varlen_support）
sageattn_1_varlen_func = None
SAGE_ATTN_1_AVAILABLE = False
try:
    try:
        from sageattention import sageattn_varlen as _sa1_varlen
    except ImportError:
        from sageattn import sageattn_varlen as _sa1_varlen
    sageattn_1_varlen_func = _sa1_varlen
    SAGE_ATTN_1_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

# 3. SageAttention 2 (varlen support)
sageattn_2_varlen_func = None
SAGE_ATTN_2_AVAILABLE = False
try:
    from sageattention import sageattn_varlen as _sa2_varlen
    sageattn_2_varlen_func = _sa2_varlen
    SAGE_ATTN_2_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

# 4. SageAttention 3 / Blackwell (RTX 50xx only, batched attention)
sageattn_blackwell = None
SAGE_ATTN_3_AVAILABLE = False
try:
    from sageattn3 import sageattn3_blackwell as _sa3_blackwell
    sageattn_blackwell = _sa3_blackwell
    SAGE_ATTN_3_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    try:
        from sageattention import sageattn_blackwell as _sa3_blackwell
        sageattn_blackwell = _sa3_blackwell
        SAGE_ATTN_3_AVAILABLE = True
    except (ImportError, AttributeError, OSError):
        pass

SAGE_ATTN_AVAILABLE = SAGE_ATTN_1_AVAILABLE or SAGE_ATTN_2_AVAILABLE or SAGE_ATTN_3_AVAILABLE


def get_best_sage_varlen():
    """Return the best available SageAttention varlen function.
    Priority: SA2 (fastest CUDA) > SA1 (Triton, works on AMD ROCm)
    Returns (func, version_str) or (None, None) if none available."""
    if SAGE_ATTN_2_AVAILABLE:
        return call_sage_attn_2_varlen, '2'
    if SAGE_ATTN_1_AVAILABLE:
        return call_sage_attn_1_varlen, '1'
    return None, None


# XB_ToolBox SageAttention preset configurations
XB_SAGE_PRESETS = {
    "XB 内置模式 A (128x128x32)": {'M': 128,  'N': 128, 'GROUP': 32, 'WAVE': 2, 'WARP': 8, 'NSTAGES': 1},
    "XB 内置模式 B (128x64x96)": {'M': 128,  'N': 64, 'GROUP': 96, 'WAVE': 3, 'WARP': 8, 'NSTAGES': 2},
    "XB 内置模式 C (128x16x16)": {'M': 128,  'N': 16, 'GROUP': 16, 'WAVE': 2, 'WARP': 4, 'NSTAGES': 2},
    "XB 内置模式 D (64x64x16)":  {'M': 64,   'N': 64, 'GROUP': 16, 'WAVE': 4, 'WARP': 4, 'NSTAGES': 2},
    "XB 自定模式 A (机智启动器)": {'M': 128, 'N': 128, 'GROUP': 32, 'WAVE': 4, 'WARP': 8, 'NSTAGES': 1},
    "XB 自定模式 B (机智启动器)": {'M': 128, 'N': 64, 'GROUP': 8, 'WAVE': 2, 'WARP': 8, 'NSTAGES': 2},
    "XB 自定模式 C (机智启动器)": {'M': 64, 'N': 64, 'GROUP': 16, 'WAVE': 1, 'WARP': 4, 'NSTAGES': 2},
}

# Check if XB_ToolBox is available (by detecting its folder in custom_nodes)
_XB_TOOLBOX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "XB_ToolBox")
XB_TOOLBOX_AVAILABLE = os.path.isdir(_XB_TOOLBOX_PATH)


def validate_attention_mode(requested_mode: str, debug=None) -> str:
    """
    Validate attention mode availability with automatic fallback.
    Now uses XB_ToolBox preset names for SageAttention modes.
    
    Args:
        requested_mode: 'sdpa' or one of XB_ToolBox preset names
        debug: Optional debug instance for logging
        
    Returns:
        Validated mode that is available
    """
    # PyTorch SDPA - always available
    if requested_mode == 'sdpa':
        return requested_mode
    
    # XB_ToolBox SageAttention presets
    if requested_mode in XB_SAGE_PRESETS:
        if not XB_TOOLBOX_AVAILABLE:
            error_msg = (
                f"Cannot use '{requested_mode}' attention mode: XB_ToolBox is not installed.\n"
                "\n"
                "XB_ToolBox provides SageAttention acceleration through optimized kernel presets.\n"
                "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
                "\n"
                "To fix this issue:\n"
                "  1. Install XB_ToolBox custom node in ComfyUI\n"
                "  2. OR change attention_mode to 'sdpa' (default, always available)\n"
            )
            if debug:
                debug.log(error_msg, level="WARNING", category="setup", force=True)
            return 'sdpa'
        
        # Verify sageattention package is installed
        if not SAGE_ATTN_AVAILABLE:
            error_msg = (
                f"Cannot use '{requested_mode}' attention mode: SageAttention package is not installed.\n"
                "\n"
                "SageAttention provides GPU acceleration for attention computations.\n"
                "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
                "\n"
                "To fix this issue:\n"
                "  1. Install SageAttention: pip install sageattention\n"
                "  2. OR change attention_mode to 'sdpa' (default, always available)\n"
                "\n"
                "For more info: https://github.com/thu-ml/SageAttention"
            )
            if debug:
                debug.log(error_msg, level="WARNING", category="setup", force=True)
            return 'sdpa'
        
        return requested_mode
    
    # Unknown mode - fallback to sdpa
    if debug:
        debug.log(
            f"Unknown attention mode '{requested_mode}'. Falling back to 'sdpa'.",
            level="WARNING", category="setup", force=True
        )
    return 'sdpa'


def get_sage_config_for_mode(attention_mode: str) -> dict:
    """
    Get SageAttention kernel config for a given attention mode.
    
    Args:
        attention_mode: Attention mode string (XB_ToolBox preset name)
        
    Returns:
        SageAttention config dict, or None if not a SageAttention mode
    """
    return XB_SAGE_PRESETS.get(attention_mode, None)

    
@torch._dynamo.disable
def call_sage_attn_1_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sage_config=None, **kwargs):
    """
    Wrapper for SageAttention 1 sageattn_varlen that handles tensor-to-scalar conversion.
    
    SageAttention 1 provides optimized int8-quantized attention for NVIDIA GPUs.
    Note: Sage1 varlen only supports head_dim 64 and 128.
    
    Args:
        sage_config: Optional XB_ToolBox SageAttention kernel config dict (M, N, GROUP, WAVE, WARP, NSTAGES)
    """
    if not SAGE_ATTN_1_AVAILABLE:
        raise ImportError("SageAttention 1 is not available")
    
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    head_dim = q.shape[-1]
    if head_dim not in [64, 128]:
        raise ValueError(f"SageAttention 1 varlen only supports head_dim 64 or 128, but got {head_dim}")

    out_dtype = q.dtype
    valid_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in valid_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    is_causal = kwargs.get('causal', False)
    smooth_k = kwargs.get('smooth_k', True)
    sm_scale = kwargs.get('sm_scale', 1.0 / (head_dim ** 0.5))
    
    # Build call args with optional sage_config from XB_ToolBox preset
    call_args = [q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k]
    call_kwargs = dict(is_causal=is_causal, sm_scale=sm_scale, smooth_k=smooth_k)
    if sage_config is not None:
        try:
            call_kwargs['sage_config'] = sage_config
        except TypeError:
            pass  # sageattn_varlen may not support sage_config in older versions
    
    out = sageattn_1_varlen_func(*call_args, **call_kwargs)
    
    return out.to(out_dtype) if out.dtype != out_dtype else out

@torch._dynamo.disable
def call_sage_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sage_config=None, **kwargs):
    """
    Wrapper for SageAttention 2 sageattn_varlen with XB_ToolBox preset support.
    
    Args:
        sage_config: Optional XB_ToolBox SageAttention kernel config dict (M, N, GROUP, WAVE, WARP, NSTAGES)
    """
    if not SAGE_ATTN_2_AVAILABLE:
        raise ImportError("SageAttention 2 is not available")
    
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    is_causal = kwargs.get('causal', False)
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    call_kwargs = {}
    if sage_config is not None:
        call_kwargs['sage_config'] = sage_config
    
    out = sageattn_2_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        is_causal, sm_scale,
        **call_kwargs
    )
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sage_attn_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sage_config=None, **kwargs):
    """
    Wrapper for SageAttention 3 (Blackwell) that converts varlen format to batched format.
    For variable-length sequences, automatically falls back to SageAttention 2.
    
    Args:
        sage_config: Optional XB_ToolBox SageAttention kernel config dict (M, N, GROUP, WAVE, WARP, NSTAGES)
    """
    if not SAGE_ATTN_3_AVAILABLE:
        raise ImportError("SageAttention 3 (Blackwell) is not available")
    
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    
    uniform_q = (seq_lens_q == seq_lens_q[0]).all()
    uniform_k = (seq_lens_k == seq_lens_k[0]).all()
    
    if not (uniform_q and uniform_k):
        if SAGE_ATTN_2_AVAILABLE:
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, sage_config=sage_config, **kwargs
            )
        raise RuntimeError(
            "SageAttention 3 (Blackwell) requires uniform sequence lengths, "
            "and SageAttention 2 is not available as fallback. "
            "Please install sageattention package or use sdpa instead."
        )
    
    # Extract batch dimensions
    batch_size = len(cu_seqlens_q) - 1
    seq_len_q = int(seq_lens_q[0].item())
    seq_len_k = int(seq_lens_k[0].item())
    heads = q.shape[1]
    dim = q.shape[2]
    
    # SageAttention requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    # Reshape varlen (total_seq, heads, dim) -> batched (batch, seq, heads, dim)
    q_batched = q.view(batch_size, seq_len_q, heads, dim)
    k_batched = k.view(batch_size, seq_len_k, heads, dim)
    v_batched = v.view(batch_size, seq_len_k, heads, dim)
    
    # SA3/Blackwell expects (batch, heads, seq, dim) layout
    q_batched = q_batched.transpose(1, 2)  # (batch, heads, seq, dim)
    k_batched = k_batched.transpose(1, 2)
    v_batched = v_batched.transpose(1, 2)
    
    # Call SA3 Blackwell
    out = sageattn_blackwell(q_batched, k_batched, v_batched, per_block_mean=False)
    
    # Reshape back to varlen format (total_seq, heads, dim)
    out = out.transpose(1, 2).reshape(-1, heads, dim).contiguous()
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


# 2. Triton - Required for torch.compile with inductor backend
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# 3. GGUF - Required for quantized model loading
try:
    import gguf
    from gguf import GGMLQuantizationType
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None
    GGMLQuantizationType = None


def validate_gguf_availability(operation: str = "load GGUF model", debug=None) -> None:
    """
    Validate GGUF availability and raise error if not installed.
    
    Args:
        operation: Description of the operation requiring GGUF
        debug: Optional debug instance for logging
        
    Raises:
        RuntimeError: If GGUF is not available
    """
    if not GGUF_AVAILABLE:
        error_msg = (
            f"Cannot {operation}: GGUF library is not installed.\n"
            f"\n"
            f"GGUF provides quantized model support for memory-efficient loading.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install GGUF: pip install gguf\n"
            f"  2. OR use a non-quantized model format (.safetensors)\n"
            f"\n"
            f"For more info: https://github.com/ggerganov/ggml"
        )
        if debug:
            debug.log(error_msg, level="ERROR", category="setup", force=True)
        raise RuntimeError(f"GGUF library required to {operation}")


# 4. NVIDIA Conv3d Memory Bug - Workaround for PyTorch >= 2.9 + cuDNN >= 91002
def _check_conv3d_memory_bug():
    """
    Check if Conv3d memory bug workaround needed.
    Bug: PyTorch 2.9+ with cuDNN >= 91002 uses 3x memory for Conv3d 
    with fp16/bfloat16 due to buggy dispatch layer.
    """
    try:
        # Exclude AMD ROCm/HIP builds (they use MIOpen, not cuDNN)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return False
        
        # Must have CUDA available
        if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            return False
        
        # Must have cuDNN actually available (not just the attribute)
        if not (hasattr(torch.backends.cudnn, 'is_available') and 
                torch.backends.cudnn.is_available()):
            return False
        
        # Check device capability (NVIDIA GPUs)
        if torch.cuda.get_device_capability()[0] < 3:
            return False
        
        # Parse torch version
        version_str = torch.__version__.split('+')[0]
        parts = version_str.split('.')
        torch_version = tuple(int(p) for p in parts[:2])
        
        # Bug affects PyTorch 2.9 and later versions
        if torch_version < (2, 9):
            return False
        
        if not hasattr(torch.backends.cudnn, 'version'):
            return False
            
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None or cudnn_version < 91002:
            return False
        
        return True
    except:
        return False

NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = _check_conv3d_memory_bug()


# Log all optimization status once globally (cross-process) using environment variable
if not os.environ.get("SEEDVR2_OPTIMIZATIONS_LOGGED"):
    os.environ["SEEDVR2_OPTIMIZATIONS_LOGGED"] = "1"
    
    # Build status strings
    sage_status = "✅" if SAGE_ATTN_AVAILABLE else "❌"
    xb_status = "✅" if XB_TOOLBOX_AVAILABLE else "❌"
    triton_status = "✅" if TRITON_AVAILABLE else "❌"
    
    # Count available optimizations
    available = [SAGE_ATTN_AVAILABLE, XB_TOOLBOX_AVAILABLE, TRITON_AVAILABLE]
    num_available = sum(available)
    
    if num_available == 3:
        print(f"⚡ SeedVR2 optimizations check: SageAttention {sage_status} | XB_ToolBox {xb_status} | Triton {triton_status}")
    elif num_available == 0:
        print(f"⚠️  SeedVR2 optimizations check: SageAttention {sage_status} | XB_ToolBox {xb_status} | Triton {triton_status}")
        print("💡 For best performance: pip install sageattention triton (XB_ToolBox for Sage presets)")
    else:
        icon = "⚡" if num_available >= 2 else "⚠️ "
        print(f"{icon} SeedVR2 optimizations check: SageAttention {sage_status} | XB_ToolBox {xb_status} | Triton {triton_status}")
        
        # Build install suggestions for missing packages
        missing = []
        if not SAGE_ATTN_AVAILABLE:
            missing.append("sageattention")
        if not TRITON_AVAILABLE:
            missing.append("triton")
        if not XB_TOOLBOX_AVAILABLE:
            missing.append("XB_ToolBox (ComfyUI custom node)")
        if missing:
            print(f"💡 Optional: install {' '.join(missing)}")
    
    # Conv3d workaround status (if applicable)
    if NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND:
        torch_ver = torch.__version__.split('+')[0]
        cudnn_ver = torch.backends.cudnn.version()
        print(f"🔧 Conv3d workaround active: PyTorch {torch_ver}, cuDNN {cudnn_ver} (fixing VAE 3x memory bug)")


# Bfloat16 CUBLAS support
def _probe_bfloat16_support() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda:0')
        _ = torch.matmul(a, a)
        del a
        return True
    except RuntimeError as e:
        if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
            return False
        raise

BFLOAT16_SUPPORTED = _probe_bfloat16_support()
COMPUTE_DTYPE = torch.bfloat16 if BFLOAT16_SUPPORTED else torch.float16


def call_rope_with_stability(method, *args, **kwargs):
    """
    Call RoPE method with stability fixes:
    1. Clear cache if available
    2. Disable autocast to prevent numerical issues (CUDA only)
    This prevents artifacts in FP8/mixed precision models.
    """
    if hasattr(method, 'cache_clear'):
        method.cache_clear()
    
    # Only use CUDA autocast context on CUDA devices
    # MPS has no CUDA autocast to disable
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=False):
            return method(*args, **kwargs)
    else:
        return method(*args, **kwargs)
    
    
class CompatibleDiT(torch.nn.Module):
    """
    Wrapper for DiT models with automatic compatibility management + advanced optimizations
    
    Precision Handling:
    - FP8: Keeps native FP8 parameters (memory efficient), converts inputs/outputs to compute_dtype for arithmetic
    - FP16/BFloat16/Float32: Uses native precision throughout
    - GGUF: On-the-fly dequantization to compute_dtype
    - MPS: Forces all parameters to compute_dtype (unified memory requires dtype consistency)
    - RoPE: Converted from FP8 to compute_dtype for numerical consistency
    
    Optimizations:
    - RoPE Stabilization: Error handling for numerical stability in mixed precision
    - MPS Compatibility: Unified dtype conversion for Apple Silicon backends
    """
    
    def __init__(self, dit_model, debug: 'Debug', compute_dtype: torch.dtype = torch.bfloat16, skip_conversion: bool = False):
        super().__init__()
        self.dit_model = dit_model
        self.debug = debug
        self.compute_dtype = compute_dtype
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16

        # Only convert if not already done (e.g., when reusing cached weights)
        if not skip_conversion and self.is_fp8_model:
            # FP8 models need RoPE frequency conversion to compute dtype
            model_variant = self._get_model_variant()
            self.debug.log(f"Detected NaDiT {model_variant} FP8 - Converting RoPE freqs for FP8 compatibility", 
                        category="precision")
            self.debug.start_timer("_convert_rope_freqs")
            self._convert_rope_freqs(target_dtype=self.compute_dtype)
            self.debug.end_timer("_convert_rope_freqs", "RoPE freqs conversion")
        
        # MPS requires unified dtype for all parameters/buffers (no autocast fallback)
        # Apply to ALL model types (FP8, FP16, GGUF) when dtype differs from compute_dtype
        if not skip_conversion and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self.model_dtype != self.compute_dtype:
                self.debug.log(f"Converting NaDiT parameters/buffers to {self.compute_dtype} for MPS backend", category="setup", force=True)
                self.debug.start_timer("_force_nadit_precision")
                self._force_nadit_precision(target_dtype=self.compute_dtype)
                self.debug.end_timer("_force_nadit_precision", "NaDiT parameters/buffers conversion")
            
        # Apply RoPE stabilization for numerical stability
        self.debug.log(f"Stabilizing RoPE computations for numerical stability", category="setup")
        self.debug.start_timer("_stabilize_rope_computations")
        self._stabilize_rope_computations()
        self.debug.end_timer("_stabilize_rope_computations", "RoPE stabilization")
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect main model dtype"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _get_model_variant(self) -> str:
        """Detect model variant from module path"""
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit_7b' in model_module:
            return "7B"
        elif 'dit_3b' in model_module:
            return "3B"
        else:
            return "Unknown"
        
    def _convert_rope_freqs(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Convert RoPE frequency buffers from FP8 to target dtype for compatibility.
        
        Args:
            target_dtype: Target dtype for RoPE freqs (default: bfloat16 for stability)
        """
        converted = 0
        for module in self.dit_model.modules():
            if 'RotaryEmbedding' in type(module).__name__:
                if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
                    if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if module.rope.freqs.device.type == "mps":
                            module.rope.freqs.data = module.rope.freqs.to("cpu").to(target_dtype).to("mps")
                        else:
                            module.rope.freqs.data = module.rope.freqs.to(target_dtype)
                        converted += 1
        self.debug.log(f"Converted {converted} RoPE frequency buffers from FP8 to {target_dtype} for compatibility", category="success")
                        
    def _force_nadit_precision(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Force ALL NaDiT parameters to target dtype to avoid promotion errors (MPS requirement).
        
        Args:
            target_dtype: Target dtype for all parameters (default: bfloat16 for MPS compatibility)
        """
        converted_count = 0
        original_dtype = None
        
        # Convert ALL parameters to target dtype
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != target_dtype:
                if param.device.type == "mps":
                    temp_cpu = param.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    param.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    param.data = param.data.to(target_dtype)
                converted_count += 1
                
        # Also convert buffers (skip GGUF quantized buffers - they have tensor_type attribute)
        for name, buffer in self.dit_model.named_buffers():
            # Skip GGUF quantized buffers - these must stay in packed format for on-the-fly dequantization
            if hasattr(buffer, 'tensor_type'):
                continue
            if buffer.dtype != target_dtype:
                if buffer.device.type == "mps":
                    temp_cpu = buffer.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    buffer.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    buffer.data = buffer.data.to(target_dtype)
                converted_count += 1
        
        self.debug.log(f"Converted {converted_count} NaDiT parameters/buffers to {target_dtype} for MPS", category="success")
        
        # Update detected dtype
        self.model_dtype = target_dtype
        self.is_fp8_model = (target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2))

    def _stabilize_rope_computations(self):
        """
        Add error handling to RoPE computations to prevent artifacts.
        
        Wraps the get_axial_freqs method of RoPE modules with a try-except handler.
        During normal operation, uses the original cached method for performance.
        Only on exceptions (e.g., numerical instability, NaN propagation) does it
        intervene by clearing the cache and retrying the computation through
        call_rope_with_stability.
        
        This prevents artifacts in FP8, mixed precision, and edge cases while
        maintaining optimal performance for normal operations.
        """
        if not hasattr(self.dit_model, 'blocks'):
            return
        
        rope_count = 0
        
        # Wrap RoPE modules to handle numerical instability
        for name, module in self.dit_model.named_modules():
            if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
                # Check if already wrapped
                if hasattr(module, '_rope_wrapped'):
                    continue
                    
                original_method = module.get_axial_freqs
                
                # Mark as wrapped and store original
                module._rope_wrapped = 'stability'
                module._original_get_axial_freqs = original_method
                
                # Error handler that prevents NaN propagation
                def stable_rope_computation(self, *args, **kwargs):
                    try:
                        return original_method(*args, **kwargs)
                    except Exception:
                        return call_rope_with_stability(original_method, *args, **kwargs)
                
                module.get_axial_freqs = types.MethodType(stable_rope_computation, module)
                rope_count += 1
        
        if rope_count > 0:
            self.debug.log(f"Stabilized {rope_count} RoPE modules", category="success")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with minimal dtype conversion overhead
        
        Conversion strategy:
            - FP16/BFloat16/Float32 models: Use native precision (no conversion needed)
            - FP8 models: Convert FP8 tensors to compute_dtype for arithmetic operations
            (FP8 parameters stay in FP8 for memory efficiency, only converted for computation)
        """
        
        # Only convert if we have an FP8 model for arithmetic operations 
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            target_dtype = self.compute_dtype
            
            # Convert args
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes:
                    converted_args.append(arg.to(target_dtype))
                else:
                    converted_args.append(arg)
            
            # Convert kwargs
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes:
                    converted_kwargs[key] = value.to(target_dtype)
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
        
        # Execute forward pass
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            self.debug.log(f"Forward pass error: {e}", level="ERROR", category="generation", force=True)
            if self.is_fp8_model:
                self.debug.log(f"FP8 model - converted FP8 tensors to {self.compute_dtype}", category="info", force=True)
            else:
                self.debug.log(f"{self.model_dtype} model - no conversion applied", category="info", force=True)
            raise
    
    def __getattr__(self, name):
        """Redirect all other attributes to original model"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Redirect assignments to original model except for our attributes"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)
                