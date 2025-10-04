# blur.py
# Fast GPU blurs (PyTorch): Gaussian (separable) and Surface Blur (bilateral, tiled to cap VRAM).
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2

__all__ = ["gaussian_blur", "surface_blur_cv"]

# -------------------- device & dtype helpers --------------------

def _pick_device(device: Optional[str]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def _to_torch_hwc(img: np.ndarray):
    arr = np.asarray(img)
    out_dtype = arr.dtype
    is_uint8 = (arr.dtype == np.uint8)

    if arr.ndim == 2:
        arr = arr[..., None]  # [H,W,1]
        gray2d = True
    elif arr.ndim == 3:
        gray2d = False
    else:
        raise ValueError(f"Expected [H,W] or [H,W,C], got {arr.shape}")

    alpha = None
    if arr.shape[-1] in (2, 4):
        alpha_np = arr[..., -1:]
        arr = arr[..., :-1]
        alpha = torch.from_numpy(alpha_np).float()

    x = torch.from_numpy(arr).float()
    vmax = 255.0 if is_uint8 else 1.0
    return x, alpha, gray2d, vmax, is_uint8, out_dtype

def _from_torch_hwc(x_hwc: torch.Tensor,
                    alpha: Optional[torch.Tensor],
                    gray2d: bool,
                    vmax: float,
                    is_uint8: bool,
                    out_dtype: np.dtype) -> np.ndarray:
    if alpha is not None:
        alpha = alpha.to(x_hwc.device)
        x_hwc = torch.cat([x_hwc, alpha], dim=-1)
    x_hwc = x_hwc.clamp_(0.0, vmax)
    if is_uint8:
        x_hwc = x_hwc.to(torch.uint8)
    out = x_hwc.cpu().numpy()
    if gray2d and out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out.astype(out_dtype, copy=False)

def _to_nchw(x_hwc: torch.Tensor) -> torch.Tensor:
    return x_hwc.permute(2, 0, 1).unsqueeze(0)

def _to_hwc(x_nchw: torch.Tensor) -> torch.Tensor:
    return x_nchw.squeeze(0).permute(1, 2, 0)

# -------------------- gaussian (separable, GPU) --------------------

def _gaussian_kernel_1d(sigma: float, radius: Optional[int]) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    g = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(g * g) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k

def gaussian_blur(
    img: np.ndarray,
    sigma: float,
    radius: Optional[int] = None,
    *,
    device: Optional[str] = None,
) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(img).copy()

    dev = _pick_device(device)
    x, alpha, gray2d, vmax, is_uint8, out_dtype = _to_torch_hwc(img)
    x = x.to(dev)
    if alpha is not None:
        alpha = alpha.to(dev)

    k1d = _gaussian_kernel_1d(sigma, radius).to(dev)
    r = (k1d.numel() - 1) // 2

    x_nchw = _to_nchw(x)                 # (1,C,H,W)
    C = x_nchw.shape[1]
    ky = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    kx = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

    y = F.conv2d(x_nchw, ky, padding=(r, 0), groups=C)
    y = F.conv2d(y,     kx, padding=(0, r), groups=C)

    y_hwc = _to_hwc(y)
    return _from_torch_hwc(y_hwc, alpha, gray2d, vmax, is_uint8, out_dtype)

# -------------------- surface blur (bilateral, tiled GPU) --------------------
def surface_blur_cv(
    img: np.ndarray,
    radius: float,
    sigma_color: float,
    sigma_space: Optional[float] = None,
    *,
    prefer_cuda: bool = True,
    border: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """
    Seam-free surface blur using OpenCV bilateral filter (GPU via cv2.cuda if available).
    Works on HxW, HxWx3, or HxWx4 NumPy arrays (uint8 or float).

    - If input is float, values are assumed in [0,1]. If sigma_color looks like 0..255, it is rescaled.
    - Alpha (4th channel) is preserved (not filtered).
    - Grayscale input is filtered as grayscale (no channel expansion in the output).
    - Returns the same dtype as input.

    Args:
        img: np.ndarray, shape (H,W), (H,W,3) or (H,W,4), dtype uint8/float32/float64
        radius: neighborhood radius in pixels (OpenCV uses diameter = 2*radius+1)
        sigma_color: range std (≈10–30 for uint8; ≈0.05–0.2 for float [0..1])
        sigma_space: spatial std in pixels; defaults to radius/2 if None
        prefer_cuda: try CUDA path if available; else CPU
        border: OpenCV border mode

    Returns:
        np.ndarray with same shape and dtype as input
    """
    if radius <= 0:
        return img.copy()

    orig_dtype = img.dtype
    if img.ndim == 2:
        H, W = img.shape
        C = 1
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        H, W, C = img.shape
    else:
        raise ValueError("img must be HxW, HxWx3, or HxWx4")

    # Split alpha if present
    alpha = None
    if C == 4:
        alpha = img[..., 3].copy()
        rgb = img[..., :3]
    elif C == 3:
        rgb = img
    else:  # grayscale
        rgb = img

    # Normalize dtype/range for processing
    is_float = np.issubdtype(rgb.dtype, np.floating)
    if is_float:
        rgb_proc = rgb.astype(np.float32, copy=False)
        # If float but looks like 0..255, convert to 0..1
        if rgb_proc.max() > 1.5:
            rgb_proc = np.clip(rgb_proc, 0, 255) / 255.0
        # If sigma_color looks like uint8 scale, rescale it
        if sigma_color > 1.0:
            sigma_color = float(sigma_color) / 255.0
    else:
        rgb_proc = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)

    # Spatial params
    if sigma_space is None:
        sigma_space = max(1e-6, radius / 2.0)
    diameter = int(max(1, 2 * int(round(radius)) + 1))

    # Prepare array(s) to feed OpenCV (BGR for color; direct for grayscale)
    if C == 1:
        src_for_cv = rgb_proc  # (H,W)
    else:
        src_for_cv = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR)  # (H,W,3)

    # Helper paths
    def cpu_bilateral(x):
        return cv2.bilateralFilter(
            x, d=diameter, sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space), borderType=border
        )

    def cuda_bilateral(x):
        # x must be contiguous
        x = np.ascontiguousarray(x)
        gsrc = cv2.cuda_GpuMat()
        gsrc.upload(x)

        # cv2.cuda.bilateralFilter works for 1ch or 3ch, uint8 or float32
        gdst = cv2.cuda.bilateralFilter(
            gsrc, d=diameter, sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space), borderType=border
        )
        out = gdst.download()
        return out

    use_cuda = False
    if prefer_cuda:
        try:
            use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            use_cuda = False

    # Some OpenCV CUDA builds might lack bilateralFilter; try/catch and fall back.
    try:
        if use_cuda:
            out_cv = cuda_bilateral(src_for_cv)
        else:
            out_cv = cpu_bilateral(src_for_cv)
    except cv2.error:
        out_cv = cpu_bilateral(src_for_cv)

    # Convert back to original channel layout
    if C == 1:
        filtered = out_cv  # (H,W)
    else:
        filtered = cv2.cvtColor(out_cv, cv2.COLOR_BGR2RGB)  # (H,W,3)

    # Restore dtype/range to match input
    if is_float:
        # Keep in [0,1] float -> cast back to the original float type
        filtered = np.clip(filtered, 0.0, 1.0).astype(orig_dtype, copy=False)
    else:
        # uint8 path
        filtered = np.clip(filtered, 0, 255).astype(orig_dtype, copy=False)

    # Reattach alpha if present
    if alpha is not None:
        if alpha.dtype != filtered.dtype:
            alpha = alpha.astype(filtered.dtype)
        filtered = np.dstack([filtered, alpha])

    return filtered