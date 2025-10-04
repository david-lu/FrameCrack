# kuwahara.py
# Fast Kuwahara filter on GPU (PyTorch), alpha-preserving, with optional jitter.
# API compatible with your NumPy version: kuwahara_fast(image_arr, radius, ...)

from __future__ import annotations
from typing import Optional, Literal, Tuple
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["kuwahara_fast"]

# ----------------------------
# Device & dtype helpers
# ----------------------------

def _pick_device(device: Optional[str]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def _to_torch_hwC(img: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, float, bool, np.dtype]:
    """
    Convert numpy [H,W] or [H,W,C] -> torch float32 [H,W,C] (no alpha),
    splitting alpha if last channel in {2,4}. Returns:
      x_hwC (float32), alpha_hw1 (float32 or None), was_gray2d, vmax, is_uint8, out_dtype
    """
    arr = np.asarray(img)
    out_dtype = arr.dtype
    is_uint8 = (arr.dtype == np.uint8)

    if arr.ndim == 2:
        arr = arr[..., None]
        was_gray2d = True
    elif arr.ndim == 3:
        was_gray2d = False
    else:
        raise ValueError(f"Expected [H,W] or [H,W,C], got {arr.shape}")

    alpha = None
    if arr.shape[-1] in (2, 4):
        alpha_np = arr[..., -1:]
        arr = arr[..., :-1]
        alpha = torch.from_numpy(alpha_np).float()

    x = torch.from_numpy(arr).float()
    vmax = 255.0 if is_uint8 else 1.0
    return x, alpha, was_gray2d, vmax, is_uint8, out_dtype

def _from_torch_hwC(
    x_hwC: torch.Tensor,
    alpha: Optional[torch.Tensor],
    was_gray2d: bool,
    vmax: float,
    is_uint8: bool,
    out_dtype: np.dtype,
) -> np.ndarray:
    """
    Torch [H,W,C] -> numpy; reattach alpha; clip in float; cast; squeeze gray.
    Safe even if x_hwC arrives as uint8/float (we always clamp in float).
    """
    # Ensure float for clipping
    if x_hwC.dtype not in (torch.float32, torch.float64):
        x_hwC = x_hwC.float()

    # Reattach alpha (as float)
    if alpha is not None:
        if alpha.dtype not in (torch.float32, torch.float64):
            alpha = alpha.float()
        if alpha.device != x_hwC.device:
            alpha = alpha.to(x_hwC.device)
        x_hwC = torch.cat([x_hwC, alpha], dim=-1)

    # Clip in float space
    x_hwC = x_hwC.clamp_(0.0, float(vmax))

    # Cast to output dtype family
    if is_uint8:
        x_hwC = x_hwC.to(torch.uint8)

    out = x_hwC.cpu().numpy()

    # Squeeze grayscale back to [H,W]
    if was_gray2d and out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]

    return out.astype(out_dtype, copy=False)

# ----------------------------
# Integral images on GPU
# ----------------------------

def _integral_image_torch(a: torch.Tensor) -> torch.Tensor:
    """
    Integral image with leading zero row/col.
    a: [H,W,C] float32
    returns S: [H+1, W+1, C] float32
    """
    s = a.cumsum(dim=0).cumsum(dim=1)                  # [H,W,C]
    H, W, C = s.shape
    out = a.new_zeros((H + 1, W + 1, C))
    out[1:, 1:, :] = s
    return out

def _rect_sum_torch(S: torch.Tensor, r1: int, c1: int, r2: int, c2: int, H: int, W: int) -> torch.Tensor:
    """
    Rectangle sum for all output pixels using integral image S.
    Returns [H,W,C].
    """
    A = S[r2 : r2 + H, c2 : c2 + W, :]
    B = S[r1 : r1 + H, c2 : c2 + W, :]
    C = S[r2 : r2 + H, c1 : c1 + W, :]
    D = S[r1 : r1 + H, c1 : c1 + W, :]
    return A - B - C + D

# ----------------------------
# Kuwahara core (GPU)
# ----------------------------

def _kuwahara_vectorized_per_channel_torch(img_hwC: torch.Tensor, r: int):
    """
    Vectorized Kuwahara over [H,W,C] using GPU integral images.
    Returns:
      out:  float32 [H,W,C] (selected quadrant mean per channel)
      qpx:  int64   [H,W,1] (quadrant index per pixel, using channel 0's min-var)
      anchors: (ar,ac) each int64 [H,W,4] top-left anchors for NW,NE,SW,SE in padded coords.
    """
    H, W, C = img_hwC.shape
    # Pad edges by r to make all quadrants valid
    padded = F.pad(img_hwC.permute(2,0,1).unsqueeze(0), (r, r, r, r), mode="replicate")  # [1,C,H+2r,W+2r]
    padded = padded.squeeze(0).permute(1,2,0).contiguous()                                # [H+2r,W+2r,C]
    sq = padded * padded

    S  = _integral_image_torch(padded)  # [Hp+1,Wp+1,C]
    S2 = _integral_image_torch(sq)

    k = float((r + 1) * (r + 1))

    # quadrant rectangle sums (top-left inclusive, bottom-right exclusive)
    sum_nw  = _rect_sum_torch(S,  0,   0,   r + 1,     r + 1,     H, W)
    sum2_nw = _rect_sum_torch(S2, 0,   0,   r + 1,     r + 1,     H, W)

    sum_ne  = _rect_sum_torch(S,  0,   r,   r + 1,     2 * r + 1, H, W)
    sum2_ne = _rect_sum_torch(S2, 0,   r,   r + 1,     2 * r + 1, H, W)

    sum_sw  = _rect_sum_torch(S,  r,   0,   2 * r + 1, r + 1,     H, W)
    sum2_sw = _rect_sum_torch(S2, r,   0,   2 * r + 1, r + 1,     H, W)

    sum_se  = _rect_sum_torch(S,  r,   r,   2 * r + 1, 2 * r + 1, H, W)
    sum2_se = _rect_sum_torch(S2, r,   r,   2 * r + 1, 2 * r + 1, H, W)

    mean_nw, var_nw = sum_nw / k,  (sum2_nw / k) - (sum_nw / k) ** 2
    mean_ne, var_ne = sum_ne / k,  (sum2_ne / k) - (sum_ne / k) ** 2
    mean_sw, var_sw = sum_sw / k,  (sum2_sw / k) - (sum_sw / k) ** 2
    mean_se, var_se = sum_se / k,  (sum2_se / k) - (sum_se / k) ** 2

    vars_  = torch.stack([var_nw,  var_ne,  var_sw,  var_se],  dim=-1)  # [H,W,C,4]
    means_ = torch.stack([mean_nw, mean_ne, mean_sw, mean_se], dim=-1)  # [H,W,C,4]

    qidx = torch.argmin(vars_, dim=-1, keepdim=True)    # [H,W,C,1]
    qpx  = qidx[..., 0, :]                               # [H,W,1] pick channel-0 choice per pixel
    out  = torch.take_along_dim(means_, qidx, dim=-1)[..., 0]  # [H,W,C]

    # anchors in padded coords relative to output (i,j)
    ii = torch.arange(H, dtype=torch.int64, device=img_hwC.device)[:, None]
    jj = torch.arange(W, dtype=torch.int64, device=img_hwC.device)[None, :]
    ar = torch.stack([ii + 0, ii + 0, ii + r, ii + r], dim=2)  # [H,W,4]
    ac = torch.stack([jj + 0, jj + r, jj + 0, jj + r], dim=2)  # [H,W,4]

    return out, qpx, (ar, ac)

# ----------------------------
# Hash-based uniform noise (GPU)
# ----------------------------

def _hashed_uniform_u01_torch(ar: torch.Tensor, ac: torch.Tensor, ch: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Deterministic uniform [0,1) from integer (ar,ac,ch), stable across runs/devices.
    Uses a 32-bit mix (mod 2**32). Shapes broadcast; returns float32.
    """
    m = (1 << 32)
    x = ( (ar.to(torch.int64) * 1315423911)
        + (ac.to(torch.int64) * 2654435761)
        + (ch.to(torch.int64) * 974593)
        + int(seed) ) % m
    # xorshift-like mixing within 32 bits
    x = x ^ (x >> 13)
    x = x ^ (x << 17)
    x = x ^ (x >> 5)
    x = x % m
    return (x.to(torch.float32) / float(m))

# ----------------------------
# Public API
# ----------------------------

def kuwahara_fast(
    image_arr: np.ndarray,
    radius: int,
    jitter_std: float = 0.0,
    seed: Optional[int] = None,
    jitter_mode: Literal["blotch", "pixel"] = "blotch",
    jitter_dist: Literal["gaussian", "uniform"] = "gaussian",
    *,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Fast Kuwahara filter (GPU if available) with optional jitter.

    Args:
      image_arr: numpy [H,W] or [H,W,C] (C in {1,2,3,4}); alpha preserved if C in {2,4}
      radius: neighborhood radius r (each quadrant is (r+1)x(r+1))
      jitter_std: jitter scale relative to value range (e.g., 0.02 ≈ ±5 on uint8)
      seed: RNG seed (affects jitter only)
      jitter_mode: 'pixel' (independent per pixel) or 'blotch' (one per chosen quadrant patch)
      jitter_dist: 'gaussian' or 'uniform'
      device: "cuda"/"cuda:0"/"cpu" (default auto)

    Returns:
      numpy array with same shape/dtype as input.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")

    dev = _pick_device(device)
    x, alpha, was_gray2d, vmax, is_uint8, out_dtype = _to_torch_hwC(image_arr)
    x = x.to(dev)
    if alpha is not None:
        alpha = alpha.to(dev)

    # Kuwahara core
    out_f32, qidx_px, anchors = _kuwahara_vectorized_per_channel_torch(x, radius)  # [H,W,C], [H,W,1]

    # ----- Optional jitter -----
    if jitter_std > 0:
        scale = float(jitter_std) * float(vmax)
        H, W, C = out_f32.shape

        if jitter_mode == "pixel":
            g = torch.Generator(device=dev)
            if seed is not None:
                g.manual_seed(int(seed))
            if jitter_dist == "gaussian":
                noise = torch.randn_like(out_f32, generator=g) * scale
            else:
                noise = (torch.rand_like(out_f32, generator=g) * 2.0 - 1.0) * scale
            out_f32 = out_f32 + noise
        else:
            # 'blotch': one noise per chosen quadrant patch (stable via hashing anchors)
            ar, ac = anchors  # [H,W,4] each
            ar_sel = torch.take_along_dim(ar, qidx_px, dim=2)[..., 0]  # [H,W]
            ac_sel = torch.take_along_dim(ac, qidx_px, dim=2)[..., 0]  # [H,W]
            arC = ar_sel[..., None].expand(H, W, C)
            acC = ac_sel[..., None].expand(H, W, C)
            ch  = torch.arange(C, device=dev, dtype=torch.int64).view(1,1,C).expand(H, W, C)

            u = _hashed_uniform_u01_torch(arC, acC, ch, seed=0 if seed is None else int(seed))
            if jitter_dist == "gaussian":
                u2 = _hashed_uniform_u01_torch(arC, acC, ch + 1337, seed=(123456789 ^ (0 if seed is None else int(seed))))
                eps = torch.finfo(torch.float32).eps
                z = torch.sqrt(-2.0 * torch.log(torch.clamp(u, min=eps, max=1.0))) * torch.cos(2.0 * torch.pi * u2)
                noise = z * scale
            else:
                noise = (u * 2.0 - 1.0) * scale
            out_f32 = out_f32 + noise

    # Reattach alpha and clip/cast back
    y = out_f32
    if alpha is not None:
        y = torch.cat([y, alpha], dim=-1)

    # Delegate clipping/casting/squeeze to the robust helper
    out = _from_torch_hwC(y, None, was_gray2d, vmax, is_uint8, out_dtype)
    return out

# -------------- quick test --------------
if __name__ == "__main__":
    # Demo gradient image
    H, W = 180, 240
    x = np.zeros((H, W, 3), dtype=np.uint8)
    x[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    x[..., 1] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]
    y = kuwahara_fast(x, radius=4, jitter_std=0.03, seed=42, jitter_mode="blotch")
    print("Output:", y.shape, y.dtype, "CUDA:", torch.cuda.is_available())
