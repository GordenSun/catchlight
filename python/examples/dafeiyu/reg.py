"""Layer loading + registration of redrawn layers back onto the base illustration.

A layer pixel (x, y) lands on base pixel (sx*x + tx, sy*y + ty)."""
import numpy as np
from PIL import Image

BG = np.array([0.0, 1.0, 0.0], np.float32)  # stand-in colour outside the base silhouette


def load(path):
    """RGBA float32 0..1 with the generator's 0..254 alpha stretched to 0..1."""
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    a[..., 3] = np.clip((a[..., 3] - 0.03) / (0.96 - 0.03), 0, 1)
    return a


def bbox(alpha, thr=0.05):
    ys, xs = np.nonzero(alpha > thr)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def split_lr(img, thr=0.05):
    """Split a two-component layer at the widest empty column gap -> (left, right) images."""
    cols = (img[..., 3] > thr).any(axis=0)
    x0, _, x1, _ = bbox(img[..., 3], thr)
    best, run, start = (0, 0), 0, None
    for x in range(x0, x1):
        if not cols[x]:
            if start is None: start = x
            run = x - start + 1
            if run > best[0]: best = (run, start)
        else:
            start = None
    if best[0] == 0: raise ValueError("no gap between the two halves")
    mid = best[1] + best[0] // 2
    l, r = img.copy(), img.copy()
    l[:, mid:, 3] = 0; r[:, :mid, 3] = 0
    return l, r


def _resize(img, w, h):
    """Alpha-aware resize of float RGBA."""
    pm = img.copy(); pm[..., :3] *= pm[..., 3:4]
    chans = [np.asarray(Image.fromarray(pm[..., c]).resize((w, h), Image.BOX if w < img.shape[1] else Image.BILINEAR)) for c in range(4)]
    out = np.stack(chans, -1)
    out[..., :3] /= np.maximum(out[..., 3:4], 1e-4)
    return out


def _on_bg(img):
    return img[..., :3] * img[..., 3:4] + BG * (1 - img[..., 3:4])


_BASE_CACHE = {}


def _base_fft(base, work):
    key = (id(base), work)
    if key not in _BASE_CACHE:
        f = work / max(base.shape[:2])
        Bs = _on_bg(_resize(base, round(base.shape[1] * f), round(base.shape[0] * f)))
        h, w = Bs.shape[:2]
        P = (2 * h, 2 * w)
        Bp = np.broadcast_to(BG, P + (3,)).copy(); Bp[h // 2:h // 2 + h, w // 2:w // 2 + w] = Bs
        _BASE_CACHE[key] = (f, P, (h // 2, w // 2), [np.fft.rfft2(Bp[..., c]) for c in range(3)], [np.fft.rfft2(Bp[..., c] ** 2) for c in range(3)])
    return _BASE_CACHE[key]


def _coarse(layer, base, scales, work=360, window=None):
    """Masked zero-mean normalised cross-correlation over every translation, per scale.

    Zero-mean, so a flat patch earns nothing for landing on a flat region and the
    scale is not pulled toward shrinking the layer into one."""
    f, P, (oy, ox), FB, FB2 = _base_fft(base, work)
    x0, y0, x1, y1 = bbox(layer[..., 3])
    crop = layer[y0:y1, x0:x1]
    out = []
    for s in scales:
        w, h = round(crop.shape[1] * f * s), round(crop.shape[0] * f * s)
        if w < 6 or h < 6 or w > P[1] // 2 or h > P[0] // 2: continue
        L = _resize(crop, w, h)
        a = L[..., 3]; n = a.sum()
        Fa = np.conj(np.fft.rfft2(a, P))
        score = np.zeros(P, np.float32)
        for c in range(3):
            l = L[..., c]; ml = (a * l).sum() / n
            varl = (a * (l - ml) ** 2).sum()
            sB = np.fft.irfft2(Fa * FB[c], P); sB2 = np.fft.irfft2(Fa * FB2[c], P)
            sLB = np.fft.irfft2(np.conj(np.fft.rfft2(a * l, P)) * FB[c], P)
            num = sLB - ml * sB
            den = np.sqrt(np.maximum(varl * (sB2 - sB ** 2 / n), 1e-6))
            score += np.where(den > 1e-2 * n, num / den, 0)
        score = score[:P[0] - h + 1, :P[1] - w + 1]
        if window is not None:
            # keep translations whose art centre lands within r of (cx, cy), in base px
            wx, wy, wr = window
            yy, xx = np.ogrid[:score.shape[0], :score.shape[1]]
            far = ((xx - ox + w / 2) / f - wx) ** 2 + ((yy - oy + h / 2) / f - wy) ** 2 > wr * wr
            score = np.where(far, -9.0, score)
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        tx = (ix - ox) / f - s * x0; ty = (iy - oy) / f - s * y0
        out.append((-float(score[iy, ix]) / 3, s, tx, ty))
    return sorted(out)


def _robust(params, pts, cols, B, tau):
    sx, sy, tx, ty = params
    x = sx * pts[:, 0] + tx; y = sy * pts[:, 1] + ty
    H, W = B.shape[:2]
    inside = (x >= 0) & (x <= W - 1.001) & (y >= 0) & (y <= H - 1.001)
    xi = np.clip(x, 0, W - 1.001); yi = np.clip(y, 0, H - 1.001)
    x0 = xi.astype(int); y0 = yi.astype(int); fx = (xi - x0)[:, None]; fy = (yi - y0)[:, None]
    s = B[y0, x0] * (1 - fx) * (1 - fy) + B[y0, x0 + 1] * fx * (1 - fy) + B[y0 + 1, x0] * (1 - fx) * fy + B[y0 + 1, x0 + 1] * fx * fy
    d = ((s - cols) ** 2).sum(1)
    d = np.where(inside, np.minimum(d, tau * tau), tau * tau)
    return float(d.mean())


def _samples(layer, n, rng):
    """Half uniform over the art, half weighted toward its lines and silhouette."""
    pm = layer[..., :3] * layer[..., 3:4]
    g = np.zeros(layer.shape[:2], np.float32)
    for c in range(3):
        gy, gx = np.gradient(pm[..., c]); g += np.abs(gx) + np.abs(gy)
    gy, gx = np.gradient(layer[..., 3]); g += np.abs(gx) + np.abs(gy)
    solid = layer[..., 3] > 0.6
    ys, xs = np.nonzero(solid)
    wts = g[ys, xs] + 1e-6
    e = rng.choice(len(xs), min(n // 2, len(xs)), replace=False, p=wts / wts.sum())
    u = rng.choice(len(xs), min(n // 2, len(xs)), replace=False)
    pick = np.concatenate([e, u])
    return np.stack([xs[pick], ys[pick]], 1).astype(np.float32), layer[ys[pick], xs[pick], :3]


def register(layer, base, scales=None, aniso=0.05, drift=0.07, tau=0.22, n=40000, seed=0, verbose="", window=None, init=None, keep=None):
    """Best (sx, sy, tx, ty) placing `layer` on `base`; both float RGBA arrays (any sizes).

    `drift` bounds how far the refinement may move the scale from the coarse
    match and `aniso` how far sx/sy may leave 1, so occluded flat art cannot
    improve its score by collapsing."""
    if scales is None: scales = np.geomspace(0.2, 1.6, 64)
    pts, cols = _samples(layer, n, np.random.default_rng(seed))
    if keep is not None:  # only score the art where keep(x, y) holds, e.g. the part known to show in the base
        m = keep(pts[:, 0], pts[:, 1]); pts, cols = pts[m], cols[m]
    B = _on_bg(base)
    if init is not None:
        cands = [(0.0, init[0], init[2], init[3])]
    else:
        # every scale's best translation, re-ranked by the occlusion-tolerant cost before refining
        cands = _coarse(layer, base, scales, window=window)
        cands = sorted(cands, key=lambda c: _robust((c[1], c[1], c[2], c[3]), pts, cols, B, tau))[:4]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    best = None
    for c0, s, tx, ty in cands:
        p = np.array([s, s, tx, ty], np.float64)
        cur = _robust(p, pts, cols, B, tau)
        steps = np.array([0.01, 0.01, 0.01 * ext * s, 0.01 * ext * s])
        for _ in range(80):
            improved = False
            for idx in [(0, 1), (2,), (3,), (0,), (1,)]:
                for sign in (1, -1):
                    q = p.copy()
                    for i in idx: q[i] += sign * steps[i]
                    if abs(q[0] / q[1] - 1) > aniso or abs(np.sqrt(q[0] * q[1]) / s - 1) > drift: continue
                    q[2] += (p[0] - q[0]) * cx; q[3] += (p[1] - q[1]) * cy  # scale about the art's centre
                    v = _robust(q, pts, cols, B, tau)
                    if v < cur - 1e-7: p, cur, improved = q, v, True
            if not improved:
                steps *= 0.5
                if steps[2] < 0.1: break
        if best is None or cur < best[0]: best = (cur, p, -c0)
    if verbose: print(f"{verbose}: ncc {best[2]:.3f} cost {best[0]:.4f} sx {best[1][0]:.4f} sy {best[1][1]:.4f} tx {best[1][2]:.1f} ty {best[1][3]:.1f}", flush=True)
    return tuple(float(v) for v in best[1])


def place(layer, tf, size):
    """Resample `layer` through `tf` onto a (w, h) canvas (float RGBA)."""
    sx, sy, tx, ty = tf
    pm = layer.copy(); pm[..., :3] *= pm[..., 3:4]
    chans = []
    for c in range(4):
        im = Image.fromarray(pm[..., c])
        chans.append(np.asarray(im.transform(size, Image.AFFINE, (1 / sx, 0, -tx / sx, 0, 1 / sy, -ty / sy), Image.BICUBIC)))
    out = np.clip(np.stack(chans, -1), 0, 1)
    out[..., :3] /= np.maximum(out[..., 3:4], 1e-4)
    return np.clip(out, 0, 1)


def over(dst, src):
    a = src[..., 3:4]
    out = dst.copy()
    out[..., :3] = src[..., :3] * a + dst[..., :3] * dst[..., 3:4] * (1 - a)
    out[..., 3:4] = a + dst[..., 3:4] * (1 - a)
    out[..., :3] /= np.maximum(out[..., 3:4], 1e-4)
    return out


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def fit_iou(mask, target, scales, work=420, window=None, verbose=""):
    """(s, tx, ty) placing boolean `mask` on boolean `target` for the best overlap (IoU)."""
    f = work / max(target.shape)
    def small(m, w, h): return np.asarray(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BOX))
    T = small(target, round(target.shape[1] * f), round(target.shape[0] * f))
    h0, w0 = T.shape
    P = (2 * h0, 2 * w0)
    Tp = np.zeros(P, np.float32); Tp[h0 // 2:h0 // 2 + h0, w0 // 2:w0 // 2 + w0] = T
    FT = np.fft.rfft2(Tp); nT = T.sum()
    ys, xs = np.nonzero(mask); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    best = None
    for s in scales:
        w, h = round(crop.shape[1] * f * s), round(crop.shape[0] * f * s)
        if w < 4 or h < 4 or w > w0 or h > h0: continue
        L = small(crop, w, h)
        inter = np.fft.irfft2(np.conj(np.fft.rfft2(L, P)) * FT, P)[:P[0] - h + 1, :P[1] - w + 1]
        iou = inter / (L.sum() + nT - inter)
        if window is not None:
            wx, wy, wr = window
            yy, xx = np.ogrid[:iou.shape[0], :iou.shape[1]]
            iou = np.where(((xx - w0 // 2 + w / 2) / f - wx) ** 2 + ((yy - h0 // 2 + h / 2) / f - wy) ** 2 > wr * wr, -1, iou)
        iy, ix = np.unravel_index(np.argmax(iou), iou.shape)
        if best is None or iou[iy, ix] > best[0]:
            best = (float(iou[iy, ix]), float(s), (ix - w0 // 2) / f - s * x0, (iy - h0 // 2) / f - s * y0)
    if verbose: print(f"{verbose}: iou {best[0]:.3f} s {best[1]:.4f} tx {best[2]:.1f} ty {best[3]:.1f}", flush=True)
    return best[1], best[1], best[2], best[3]


def tf_bbox(img_or_mask, tf, thr=0.05):
    """A layer's solid bounding box in base pixels after `tf`."""
    a = img_or_mask[..., 3] if img_or_mask.ndim == 3 else img_or_mask
    x0, y0, x1, y1 = bbox(a, thr)
    sx, sy, tx, ty = tf
    return sx * x0 + tx, sy * y0 + ty, sx * x1 + tx, sy * y1 + ty


def fit_cover(mask, target, forbid, scales, lam=1.0, work=420, verbose=""):
    """(s, tx, ty) placing `mask` to cover `target` while keeping off `forbid`.

    Unlike IoU, art that lands on neither (hidden behind other parts) is free."""
    f = work / max(target.shape)
    def small(m, w, h): return np.asarray(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BOX))
    h0, w0 = round(target.shape[0] * f), round(target.shape[1] * f)
    P = (2 * h0, 2 * w0)
    def padded(m, fill):
        p = np.full(P, fill, np.float32); p[h0 // 2:h0 // 2 + h0, w0 // 2:w0 // 2 + w0] = small(m, w0, h0); return np.fft.rfft2(p)
    FT, FF = padded(target, 0.0), padded(forbid, 1.0)
    ys, xs = np.nonzero(mask); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    best = None
    for s in scales:
        w, h = round(crop.shape[1] * f * s), round(crop.shape[0] * f * s)
        if w < 4 or h < 4 or w > w0 or h > h0: continue
        FL = np.conj(np.fft.rfft2(small(crop, w, h), P))
        score = (np.fft.irfft2(FL * FT, P) - lam * np.fft.irfft2(FL * FF, P))[:P[0] - h + 1, :P[1] - w + 1]
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        if best is None or score[iy, ix] > best[0]:
            best = (float(score[iy, ix]), float(s), (ix - w0 // 2) / f - s * x0, (iy - h0 // 2) / f - s * y0)
    if verbose: print(f"{verbose}: cover {best[0]:.0f} s {best[1]:.4f} tx {best[2]:.1f} ty {best[3]:.1f}", flush=True)
    return best[1], best[1], best[2], best[3]


def blur(a, r):
    """Approximate Gaussian blur of a 2D float array (three box passes of radius r)."""
    r = max(1, int(round(r)))
    out = a.astype(np.float32)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]; pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis); lo = np.take(c, np.arange(0, n), axis=axis)
            out = (hi - lo) / (2 * r + 1)
    return out
