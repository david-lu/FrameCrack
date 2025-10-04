import cv2
import numpy as np
from typing import Optional

# =======================
#     CUDA + helpers
# =======================


def _probe_cuda() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_CUDA_AVAILABLE = _probe_cuda()
_cuda_cache = {}  # (low,high,l2)-> cv2.cuda_CannyEdgeDetector


def _get_cuda_canny(low: float, high: float, l2: bool):
    key = (float(low), float(high), bool(l2))
    det = _cuda_cache.get(key)
    if det is None:
        det = cv2.cuda.createCannyEdgeDetector(low, high, L2gradient=l2)
        _cuda_cache[key] = det
    return det


def _to_gpumat(a: np.ndarray):
    g = cv2.cuda_GpuMat()
    g.upload(np.ascontiguousarray(a))
    return g


def _ensure_u8(x: np.ndarray) -> np.ndarray:
    """Return uint8; assume float is [0,1] if max<=1.5, else clamp 0..255."""
    if x.dtype is np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        if x.size and x.max() <= 1.5:
            x = np.clip(x, 0.0, 1.0) * 255.0
        else:
            x = np.clip(x, 0.0, 255.0)
    else:
        x = np.clip(x, 0, 255)
    return x.astype(np.uint8, copy=False)


def _gray_u8(x: np.ndarray) -> np.ndarray:
    """Fast luminance to uint8."""
    x = _ensure_u8(x)
    if x.ndim == 2:
        return x
    c = x.shape[2]
    if c == 1:
        return x[..., 0]
    if c == 3:
        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    if c == 4:
        return cv2.cvtColor(x, cv2.COLOR_RGBA2GRAY)
    raise ValueError("Unsupported channel count")


def _rgb3_u8(x: np.ndarray) -> np.ndarray:
    """Return RGB uint8 (drop alpha if present)."""
    x = _ensure_u8(x)
    if x.ndim == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    if x.shape[2] == 3:
        return x
    if x.shape[2] == 4:
        return x[..., :3]
    raise ValueError("Unsupported channel count")


# =======================
#      NumPy APIs
# =======================


def canny_edge_detect_cv(
    img: np.ndarray,
    low: float,
    high: float,
    *,
    use_rgb: bool = False,
    l2gradient: bool = True,
    prefer_cuda: bool = True,
) -> np.ndarray:
    """
    Fast Canny with optional CUDA acceleration.
    Returns uint8 edge map (H,W). If use_rgb=True, ORs edges from R,G,B.
    """
    use_cuda = prefer_cuda and _CUDA_AVAILABLE

    if not use_rgb:
        gray = _gray_u8(img)
        if use_cuda:
            det = _get_cuda_canny(low, high, l2gradient)
            e = det.detect(_to_gpumat(gray))
            return e.download()
        return cv2.Canny(gray, low, high, L2gradient=l2gradient)

    # use_rgb=True
    rgb = _rgb3_u8(img)
    if use_cuda:
        g_rgb = _to_gpumat(rgb)
        r, g, b = cv2.cuda.split(g_rgb)
        det = _get_cuda_canny(low, high, l2gradient)
        er = det.detect(r)
        eg = det.detect(g)
        eb = det.detect(b)
        out = cv2.cuda.bitwise_or(cv2.cuda.bitwise_or(er, eg), eb)
        return out.download()

    r, g, b = cv2.split(rgb)
    out = cv2.bitwise_or(
        cv2.Canny(r, low, high, L2gradient=l2gradient),
        cv2.Canny(g, low, high, L2gradient=l2gradient),
    )
    out = cv2.bitwise_or(out, cv2.Canny(b, low, high, L2gradient=l2gradient))
    return out


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
    Edge-preserving bilateral (full-frame, seam-free).
    Preserves alpha if present (RGB filtered, A copied).
    """
    if sigma_space is None:
        sigma_space = max(1e-6, radius / 2.0)
    d = int(max(1, 2 * int(round(radius)) + 1))

    # split alpha if any
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        base = img[..., :3]
    else:
        base = img

    bgr = cv2.cvtColor(_rgb3_u8(base), cv2.COLOR_RGB2BGR)

    use_cuda = prefer_cuda and _CUDA_AVAILABLE
    if use_cuda:
        gdst = cv2.cuda.bilateralFilter(
            _to_gpumat(bgr),
            d=d,
            sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space),
            borderType=border,
        )
        bgr_blur = gdst.download()
    else:
        bgr_blur = cv2.bilateralFilter(
            bgr,
            d=d,
            sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space),
            borderType=border,
        )
    rgb_blur = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2RGB)

    if alpha is not None:
        if alpha.dtype != rgb_blur.dtype:
            alpha = alpha.astype(rgb_blur.dtype)
        rgb_blur = np.dstack([rgb_blur, alpha])

    # match input dtype/range
    if np.issubdtype(img.dtype, np.floating):
        return (rgb_blur.astype(np.float32) / 255.0).astype(img.dtype)
    return rgb_blur.astype(img.dtype, copy=False)


def gaussian_blur_cv(
    img: np.ndarray,
    blur: float,  # Pillow-compatible: blur == sigma (pixels)
    *,
    prefer_cuda: bool = True,
    border: int = cv2.BORDER_REFLECT101,
) -> np.ndarray:
    """
    Gaussian blur with Pillow-compatible API: `blur` == sigma in pixels.
    - If blur <= 0: returns a copy (no-op).
    - Preserves alpha (blurs RGB/gray only).
    - CUDA when available, CPU otherwise.
    """
    if blur <= 0:
        return img.copy()

    # Split alpha if present
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        base = img[..., :3]
    else:
        base = img

    # Work dtype: keep float as float32 for precision; keep uint8 as is
    orig_dtype = base.dtype
    work = (
        base.astype(np.float32, copy=False)
        if np.issubdtype(orig_dtype, np.floating)
        else base
    )

    # Choose odd kernel size covering ~±3σ (matches Pillow behavior)
    k = int(np.ceil(blur * 6.0)) | 1
    k = max(k, 3)
    ksize = (k, k)

    is_gray = (work.ndim == 2) or (work.ndim == 3 and work.shape[2] == 1)
    use_cuda = prefer_cuda and _CUDA_AVAILABLE

    if use_cuda:
        # Map dtype/channels to OpenCV type for CUDA filter
        ch = 1 if is_gray else 3
        if is_gray and work.ndim == 3:
            work = work[..., 0]
        if np.issubdtype(work.dtype, np.floating):
            src_type = cv2.CV_32FC1 if ch == 1 else cv2.CV_32FC3
        else:
            src_type = cv2.CV_8UC1 if ch == 1 else cv2.CV_8UC3

        gf = cv2.cuda.createGaussianFilter(
            src_type,
            src_type,
            ksize,
            sigmaX=float(blur),
            sigmaY=float(blur),
            borderMode=border,
        )
        out = gf.apply(_to_gpumat(work)).download()
        if not is_gray and out.ndim == 2:
            out = np.stack([out, out, out], axis=-1)
    else:
        out = cv2.GaussianBlur(
            work, ksize=ksize, sigmaX=float(blur), sigmaY=float(blur), borderType=border
        )

    # Reattach alpha
    if alpha is not None:
        if alpha.dtype != out.dtype:
            alpha = alpha.astype(out.dtype)
        out = np.dstack([out, alpha])

    # Cast back to original dtype
    if out.dtype != orig_dtype:
        if np.issubdtype(orig_dtype, np.floating):
            out = out.astype(orig_dtype, copy=False)
        else:
            out = np.clip(out, 0, 255).astype(orig_dtype, copy=False)

    return out
