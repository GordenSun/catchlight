"""align.py <q|adult> [layer ...]: register the redrawn layers onto the base illustration.

Writes <ch>/transforms.json (layer px -> base px, as sx, sy, tx, ty) and a preview.
Named layers are re-registered; without names everything is."""
import json, os, sys
import numpy as np
from PIL import Image
import reg, layers

ch = sys.argv[1]
want = set(sys.argv[2:])
CFG = {
    "q": dict(tail_roi=(0.55, 0.52), tail_scales=(0.6, 0.72), tail_lam=0.6, hair_scales=(0.6, 1.1), head_s0=1.12),
    "adult": dict(tail_roi=(0.60, 0.42), tail_scales=(0.25, 0.6), tail_lam=0.8, hair_scales=(0.6, 1.2), head_s0=1.12),
}[ch]
base = reg.load(f"{ch}/base.png"); H, W = base.shape[:2]
r, g, b, a = (base[..., i] for i in range(4))
path = f"{ch}/transforms.json"
tf = json.load(open(path)) if os.path.exists(path) else {}
src = layers.load_all(ch)
def todo(name): return name not in tf or name in want or (not want and "--keep" not in sys.argv)
def mask(name): return reg.place(src[name], tf[name], (W, H))[..., 3] > 0.5

# 1. layers that show mostly unoccluded in the base: straight colour registration
for name in ["fronthair", "bodyna", "fins_l", "fins_r", "arms_l", "arms_r"]:
    # a redrawn body can come back with different proportions, so it may stretch more than the rest
    if todo(name): tf[name] = reg.register(src[name], base, verbose=name, aniso=0.15 if name == "bodyna" else 0.05)

# 2. face-canvas layers register against the face crop, then map into base pixels
box = json.load(open(f"{ch}/face_crop.json"))["box"]
# a sharp redraw of the same crop, when there is one, is the better thing to match fine line art against
crop_file = f"{ch}/face_hd.png" if os.path.exists(f"{ch}/face_hd.png") else f"{ch}/face_crop.png"
crop = np.asarray(Image.open(crop_file).convert("RGBA").resize((2048, 2048), Image.LANCZOS)).astype(np.float32) / 255
k = (box[2] - box[0]) / crop.shape[1]
def face_reg(name, img, scales=(0.4, 1.1), **kw):
    sx, sy, tx, ty = reg.register(img, crop, scales=np.geomspace(*scales, 40), verbose=name, **kw)
    return (sx * k, sy * k, box[0] + tx * k, box[1] + ty * k)
for side in "lr":
    if todo(f"iris_{side}"): tf[f"iris_{side}"] = face_reg(f"iris_{side}", src[f"iris_{side}"])
    # the lash layer splits into the upper lash (largest piece) and the lower lash line
    src[f"lashup_{side}"], src[f"lashlow_{side}"] = layers.split_lash(src[f"lash_{side}"])
for side in "lr":
    # the whole lash layer registers on its heavy upper lash; the lower line is re-seated below
    if todo(f"lashup_{side}"): tf[f"lashup_{side}"] = face_reg(f"lashup_{side}", src[f"lash_{side}"])
for side in "lr":
    up, low = src[f"lashup_{side}"], src[f"lashlow_{side}"]
    if todo(f"lashlow_{side}"):
        # lower lash: same scale as the upper one, centred under the iris, hugging its bottom edge
        sx, sy, _, _ = tf[f"lashup_{side}"]
        ix0, iy0, ix1, iy1 = reg.tf_bbox(src[f"iris_{side}"], tf[f"iris_{side}"])
        lx0, ly0, lx1, ly1 = reg.bbox(low[..., 3])
        cx = np.average(np.arange(low.shape[1]), weights=low[..., 3].sum(0))  # the line's own centre of mass
        # the line's height is its median over the central columns: corner pieces and hanging lashes do not pull it
        mids = [np.average(np.arange(low.shape[0]), weights=low[:, x, 3]) for x in range(int(cx - 0.25 * (lx1 - lx0)), int(cx + 0.25 * (lx1 - lx0))) if low[:, x, 3].sum() > 0.5]
        cy = float(np.median(mids))
        tf[f"lashlow_{side}"] = (sx, sy, (ix0 + ix1) / 2 - sx * cx, iy1 - 0.01 * (iy1 - iy0) - sy * cy)
    if todo(f"brows_{side}"):
        # brows hide under the bangs: search just above the eye (in crop px), at about the lash line's width
        ux0, uy0, ux1, uy1 = [(v - box[i % 2]) / k for i, v in enumerate(reg.tf_bbox(up, tf[f"lashup_{side}"]))]
        bm = src[f"brows_{side}"]; bx0, _, bx1, _ = reg.bbox(bm[..., 3])
        s1 = (ux1 - ux0) / (bx1 - bx0)
        tf[f"brows_{side}"] = face_reg(f"brows_{side}", bm, scales=(s1 * 0.6, s1 * 1.1),
                                       window=((ux0 + ux1) / 2, uy0 - 0.3 * (uy1 - uy0), 0.5 * (ux1 - ux0)))

# 3. tail: cover the base's tail-coloured pixels without spilling past its silhouette
if todo("tail"):
    roi = np.zeros((H, W), bool); roi[int(CFG["tail_roi"][1] * H):, int(CFG["tail_roi"][0] * W):] = True
    tailc = roi & (a > 0.5) & (b - r > 0.2) & (b > 0.38) & (b < 0.82) & (g < 0.5)
    tf["tail"] = reg.fit_cover(src["tail"][..., 3] > 0.5, tailc, a < 0.5, np.geomspace(*CFG["tail_scales"], 24), lam=CFG["tail_lam"], verbose="tail")

# 4. back hair: overlap the base's hair-blue pixels (tail excluded)
if todo("backhair"):
    hair = (a > 0.5) & (b - r > 0.28) & (b > 0.5) & ~mask("tail")
    tf["backhair"] = reg.fit_iou(src["backhair"][..., 3] > 0.5, hair, np.geomspace(*CFG["hair_scales"], 40), verbose="backhair")

# 5. head: its jaw outline on the base's, while covering every visible face-skin pixel
if todo("head"):
    skin = (a > 0.8) & (r > 0.92) & (g > 0.8) & (g < 0.96) & (b > 0.72) & (b < 0.93) & (r - b > 0.05)
    # the face is the skin connected to a point between the eyes: the jaw line fences the neck off
    il, ir = reg.tf_bbox(src["iris_l"], tf["iris_l"]), reg.tf_bbox(src["iris_r"], tf["iris_r"])
    ex, ey, ed = (il[0] + il[2] + ir[0] + ir[2]) / 4, (il[1] + il[3] + ir[1] + ir[3]) / 4, (ir[0] + ir[2] - il[0] - il[2]) / 2
    d = max(1, int(ed / 80))
    hh, ww = H // d * d, W // d * d
    small = skin[:hh, :ww].reshape(hh // d, d, ww // d, d).all(axis=(1, 3))   # a block counts only if all skin, so thin lines still fence
    lab, _ = layers.components(small)
    seed = lab[int((ey + 0.55 * ed) / d), int(ex / d)]
    assert seed, "no face skin under the eyes"
    face = np.zeros((H, W), bool); face[:hh, :ww] = np.repeat(np.repeat(lab == seed, d, 0), d, 1)
    reg.save(np.dstack([face, face, face, np.ones_like(face)]).astype(np.float32)[::4, ::4], f"prev/{ch}_facemask.png")
    ys, xs = np.nonzero(face)
    sx0, sx1, sy1 = np.percentile(xs, 0.5), np.percentile(xs, 99.5), np.percentile(ys, 99.8)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(xs), min(8000, len(xs)), replace=False); fpts = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)
    hd = src["head"]; ha = hd[..., 3]
    hx0, hy0, hx1, hy1 = reg.bbox(ha)
    # the jaw half of the head: wherever it lands should be face, or hair hanging over the face
    jy, jx = np.nonzero(ha > 0.5); keep = jy > hy0 + 0.5 * (hy1 - hy0); jy, jx = jy[keep], jx[keep]
    pick = rng.choice(len(jx), min(12000, len(jx)), replace=False); jpts = np.stack([jx[pick], jy[pick]], 1).astype(np.float32)
    allowed = face | ((a > 0.5) & (b - r > 0.12))
    def cost(p):
        s, tx, ty = p
        x = np.clip(s * jpts[:, 0] + tx, 0, W - 1).astype(int); y = np.clip(s * jpts[:, 1] + ty, 0, H - 1).astype(int)
        spill = (~allowed[y, x]).mean()
        lx = (fpts[:, 0] - tx) / s; ly = (fpts[:, 1] - ty) / s
        ok = (lx >= 0) & (lx < ha.shape[1]) & (ly >= 0) & (ly < ha.shape[0])
        cov = np.zeros(len(lx), bool); cov[ok] = ha[ly[ok].astype(int), lx[ok].astype(int)] > 0.5
        return spill + 3.0 * (1 - cov.mean())
    s0 = (sx1 - sx0) / (hx1 - hx0)
    best = None
    for s in s0 * np.geomspace(0.85, 1.6, 16):
        for dx in np.linspace(-0.12, 0.12, 9):
            for dy in np.linspace(-0.25, 0.1, 15):
                p = np.array([s, (sx0 + sx1) / 2 - s * (hx0 + hx1) / 2 + dx * (sx1 - sx0), sy1 - s * hy1 + dy * (sx1 - sx0)])
                v = cost(p)
                if best is None or v < best[0]: best = (v, p)
    cur, p = best
    steps = np.array([0.02 * p[0], 0.02 * (sx1 - sx0), 0.02 * (sx1 - sx0)])
    while steps[1] > 0.3:
        improved = False
        for i3 in range(3):
            for sign in (1, -1):
                q = p.copy(); q[i3] += sign * steps[i3]
                if i3 == 0: q[1] += (p[0] - q[0]) * (hx0 + hx1) / 2; q[2] += (p[0] - q[0]) * hy1  # scale about the chin
                v = cost(q)
                if v < cur - 1e-6: p, cur, improved = q, v, True
        if not improved: steps *= 0.5
    print(f"head: cost {cur:.4f} s {p[0]:.4f} tx {p[1]:.1f} ty {p[2]:.1f}")
    tf["head"] = (float(p[0]), float(p[0]), float(p[1]), float(p[2]))

json.dump(tf, open(path, "w"), indent=1)
order = [n for n in layers.ORDER if not n.startswith(("lash_", "eyewhite_"))]
i = order.index("iris_r") + 1
order[i:i] = ["lashlow_l", "lashlow_r", "lashup_l", "lashup_r"]
comp = layers.composite(ch, src, tf, base, order=order)
reg.save(comp, f"{ch}/composite.png")
bx0, by0, bx1, by1 = [int(v) for v in reg.tf_bbox(src["head"], tf["head"])]
m = int(0.25 * (bx1 - bx0)); x0, y0, x1, y1 = max(0, bx0 - m), max(0, by0 - m), min(W, bx1 + m), min(H, by1 + m)
def flat(im): c = im[y0:y1, x0:x1]; return c[..., :3] * c[..., 3:4] + np.array([1, 0, 1], np.float32) * (1 - c[..., 3:4])
side = np.concatenate([flat(base), flat(comp)], 1)
Image.fromarray((side * 255).astype(np.uint8)).resize((1600, int(1600 * side.shape[0] / side.shape[1])), Image.LANCZOS).save(f"prev/{ch}_zoom_head.jpg", quality=92)
