"""Shared: load a character's source layers (split into left/right where needed)."""
import json
import numpy as np
from PIL import Image
import reg

FULL = ["backhair", "fronthair", "head", "bodyna", "tail"]
PAIRS_FULL = ["fins", "arms"]
PAIRS_FACE = ["eyewhite", "iris", "lash", "brows"]
ORDER = ["backhair", "tail", "bodyna", "arms_l", "arms_r", "fins_l", "fins_r", "head", "eyewhite_l", "eyewhite_r",
         "iris_l", "iris_r", "lash_l", "lash_r", "brows_l", "brows_r", "fronthair"]


def load_all(ch, only=None):
    src = {}
    for n in FULL:
        if only is None or n in only: src[n] = reg.load(f"{ch}/L_{n}.png")
    for n in PAIRS_FULL + PAIRS_FACE:
        if only is None or n + "_l" in only or n + "_r" in only:
            src[n + "_l"], src[n + "_r"] = reg.split_lr(reg.load(f"{ch}/{'L' if n in PAIRS_FULL else 'F'}_{n}.png"))
    return src


def composite(ch, src, tf, base, order=ORDER, out=None, h=1100):
    H, W = base.shape[:2]
    comp = np.zeros((H, W, 4), np.float32)
    for n in order:
        if n in src and n in tf: comp = reg.over(comp, reg.place(src[n], tf[n], (W, H)))
    def flat(img, bg=(1, 0, 1)): return img[..., :3] * img[..., 3:4] + np.array(bg, np.float32) * (1 - img[..., 3:4])
    side = np.concatenate([flat(base), flat(comp)], 1)
    im = Image.fromarray((side * 255).astype(np.uint8)); im.resize((im.width * h // im.height, h), Image.LANCZOS).save(out or f"prev/{ch}_composite.jpg", quality=90)
    return comp


def components(mask):
    """4-connected components of a small boolean mask -> (labels, count); 0 is background."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]: continue
        n += 1; lab[y0, x0] = n; stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; stack.append((yy, xx))
    return lab, n


def split_lash(img):
    """One eye's lash layer -> (upper lash, lower lash line), each a full-size RGBA copy."""
    x0, y0, x1, y1 = reg.bbox(img[..., 3])
    d = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / 320)))
    small = img[y0:y1:d, x0:x1:d, 3] > 0.25
    lab, n = components(small)
    areas = np.bincount(lab.ravel(), minlength=n + 1); areas[0] = 0
    top = int(np.argmax(areas))
    cy = {i: np.nonzero(lab == i)[0].mean() for i in range(1, n + 1)}
    lower = np.isin(lab, [i for i in range(1, n + 1) if i != top and cy[i] > cy[top]])
    # nearest-label upsample, then grow one cell so antialiased edges stay with their piece
    grow = lower.copy()
    grow[1:] |= lower[:-1]; grow[:-1] |= lower[1:]; grow[:, 1:] |= lower[:, :-1]; grow[:, :-1] |= lower[:, 1:]
    grow &= ~((lab > 0) & ~lower)
    big = np.repeat(np.repeat(grow, d, 0), d, 1)[:y1 - y0, :x1 - x0]
    up, low = img.copy(), img.copy()
    full = np.zeros(img.shape[:2], bool); full[y0:y1, x0:x1] = big
    # level with the middle of the opening: the widest gap down the central columns
    a = img[y0:y1, x0:x1, 3] > 0.25
    c0, c1 = int(0.4 * a.shape[1]), int(0.6 * a.shape[1])
    rows = np.nonzero(a[:, c0:c1].any(1))[0]
    g = int(np.argmax(np.diff(rows))); mid = (rows[g] + rows[g + 1]) // 2
    if (lab[mid // d:, c0 // d:c1 // d] == top).any():
        # the lower line joins the upper lash at a corner, so it is all one piece: cut it there
        full = np.zeros(img.shape[:2], bool); full[y0 + mid:] = True
    up[..., 3] = np.where(full, 0, img[..., 3]); low[..., 3] = np.where(full, img[..., 3], 0)
    return up, low
