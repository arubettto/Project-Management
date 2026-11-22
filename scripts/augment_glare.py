import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import SEED

AUG_CONFIG = {
    "glare": {                     
        "enabled": True,
        "target_regex": "glare",    
        "p": 0.7,                  
        "n_ellipses": [1, 3],
        "alpha": [150, 230],
        "size_frac": [0.15, 0.50],  
    },
    "motion_blur": {              
        "enabled": False,
        "target_regex": "blur",
        "p": 0.5,
        "kernel": [7, 21],
        "angle": [0, 180],
    },
}

_rng = np.random.default_rng(SEED)   


def add_glare(img, cfg, rng=_rng):
    h, w = img.shape[:2]
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(int(rng.integers(cfg["n_ellipses"][0], cfg["n_ellipses"][1] + 1))):
        ew = max(4, int(w * rng.uniform(*cfg["size_frac"])))
        eh = max(4, int(h * rng.uniform(*cfg["size_frac"])))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        alpha = int(rng.integers(cfg["alpha"][0], cfg["alpha"][1] + 1))
        draw.ellipse([cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2], fill=(255, 255, 255, alpha))
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def motion_blur(img, cfg, rng=_rng):
    k = int(rng.integers(cfg["kernel"][0], cfg["kernel"][1] + 1))
    angle = float(rng.uniform(*cfg["angle"]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def augment(x, is_glare, is_blur, force=False):
    """x: float32 HxWx3 in 0..255. Returns float32 in 0..255 (preprocessing is applied afterwards)."""
    img = np.clip(x, 0, 255).astype(np.uint8)
    g, b = AUG_CONFIG["glare"], AUG_CONFIG["motion_blur"]
    if is_glare and g["enabled"] and (force or _rng.random() < g["p"]):
        img = add_glare(img, g)
    if is_blur and b["enabled"] and (force or _rng.random() < b["p"]):
        img = motion_blur(img, b)
    return img.astype(np.float32)
