import numpy as np

def enhance_color(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Enhance color (saturation) of an image array using numpy.

    Args:
        img: np.ndarray of shape [H, W, 3] (RGB) or [H, W, 4] (RGBA, alpha preserved).
        factor: Saturation multiplier (1.0 = no change, >1 = more color, <1 = desaturate).

    Returns:
        np.ndarray of same shape and dtype as input, with saturation adjusted.
    """
    arr = img.astype(np.float32)

    has_alpha = arr.shape[-1] == 4
    if has_alpha:
        rgb, alpha = arr[..., :3], arr[..., 3:]
    else:
        rgb, alpha = arr, None

    # Convert to grayscale (luminance) using Rec. 601 coefficients
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]

    # Interpolate between grayscale and original RGB
    rgb_enh = lum + (rgb - lum) * factor

    # Clip back to valid range
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        rgb_enh = np.clip(rgb_enh, 0, maxval)
    else:
        rgb_enh = np.clip(rgb_enh, 0.0, 1.0)

    # Reattach alpha if present
    if has_alpha:
        out = np.concatenate([rgb_enh, alpha], axis=-1)
    else:
        out = rgb_enh

    return out.astype(img.dtype)