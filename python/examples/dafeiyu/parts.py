"""parts.py <q|adult>: cut the registered layers into placed part textures.

Writes art/<ch>/<name>.png and art/<ch>/parts.json; each entry has the part's
rectangle in base pixels (x, y, w, h) and its texel density."""
import json, os, sys
import numpy as np
from PIL import Image, ImageFilter
import reg, layers

ch = sys.argv[1]
base = reg.load(f"{ch}/base.png"); H, W = base.shape[:2]
tf = json.load(open(f"{ch}/transforms.json"))
src = layers.load_all(ch)
for side in "lr": src[f"lashup_{side}"], src[f"lashlow_{side}"] = layers.split_lash(src[f"lash_{side}"])
out = f"art/{ch}"; os.makedirs(out, exist_ok=True)   # not "parts/": the repository ignores that name
# texels per base pixel: the redrawn layers hold more detail than the base, most of all on the small adult head
FACE = {"q": 2.0, "adult": 5.0}[ch]
HEADD = {"q": 1.0, "adult": 2.0}[ch]
DENS = {n: FACE for n in ["iris_l", "iris_r", "lashup_l", "lashup_r", "lashlow_l", "lashlow_r", "brows_l", "brows_r"]}
DENS.update({n: HEADD for n in ["head", "fronthair", "fins_l", "fins_r"]})
meta = {}
marks = {"eyes": {}}


def bleed(img, rounds=6):
    """Push edge colours out under the transparent texels so filtering never pulls in black."""
    rgb, a = img[..., :3].copy(), img[..., 3]
    known = a > 0.02
    for _ in range(rounds):
        pm = np.dstack([rgb * known[..., None], known.astype(np.float32)])
        bl = np.stack([reg.blur(pm[..., c], 2) for c in range(4)], -1)
        fill = (bl[..., 3] > 1e-3) & ~known
        rgb[fill] = bl[fill][:, :3] / bl[fill][:, 3:4]
        known |= fill
    return np.dstack([rgb, a])


def render(name, d):
    """The layer resampled into its own base-pixel rectangle at density d."""
    x0, y0, x1, y1 = reg.tf_bbox(src[name], tf[name], thr=0.02)
    x0, y0, x1, y1 = int(np.floor(x0)) - 3, int(np.floor(y0)) - 3, int(np.ceil(x1)) + 3, int(np.ceil(y1)) + 3
    sx, sy, tx, ty = tf[name]
    img = reg.place(src[name], (sx * d, sy * d, (tx - x0) * d, (ty - y0) * d), (int((x1 - x0) * d), int((y1 - y0) * d)))
    return img, (x0, y0, x1 - x0, y1 - y0)


def emit(name, img, rect, d):
    img = img.copy(); img[..., 3] = np.where(img[..., 3] < 0.02, 0, img[..., 3])
    reg.save(bleed(img), f"{out}/{name}.png")
    meta[name] = {"file": f"{name}.png", "rect": [float(v) for v in rect], "density": d}


for name in ["backhair", "tail", "bodyna", "arms_l", "arms_r", "fins_l", "fins_r", "head", "iris_l", "iris_r",
             "lashlow_l", "lashlow_r", "lashup_l", "lashup_r", "brows_l", "brows_r", "fronthair"]:
    d = DENS.get(name, 1.0)
    img, rect = render(name, d)
    emit(name, img, rect, d)

# The bows sit on top of the hair while the fins tuck under it: lift the bright-blue bow out as its own part.
for side in "lr":
    d = DENS[f"fins_{side}"]
    img, rect = render(f"fins_{side}", d)
    r_, g_, b_, a_ = (img[..., i] for i in range(4))
    bow = (a_ > 0.5) & (b_ > 0.5) & (b_ - r_ > 0.25) & (g_ > 0.32)
    grow = reg.blur(bow.astype(np.float32), 5 * d) > 0.08
    solid = reg.blur(bow.astype(np.float32), 14 * d) > 0.2   # drop stray blue specks
    cut = img.copy(); cut[..., 3] = np.where(grow & solid, img[..., 3], 0)
    ys, xs_ = np.nonzero(cut[..., 3] > 0.02)
    if not len(ys): continue
    x0, y0, x1, y1 = xs_.min() - 2, ys.min() - 2, xs_.max() + 3, ys.max() + 3
    emit(f"bow_{side}", cut[y0:y1, x0:x1], (rect[0] + x0 / d, rect[1] + y0 / d, (x1 - x0) / d, (y1 - y0) / d), d)

# Eye whites: the generated eye-white shape, stretched into the opening the lashes leave, with the lid
# curves the blink rig needs read back off the result.
for side in "lr":
    d = FACE
    up, r_up = render(f"lashup_{side}", d); low, r_low = render(f"lashlow_{side}", d)
    x0 = min(r_up[0], r_low[0]); y0 = min(r_up[1], r_low[1])
    x1 = max(r_up[0] + r_up[2], r_low[0] + r_low[2]); y1 = max(r_up[1] + r_up[3], r_low[1] + r_low[3])
    w, h = int((x1 - x0) * d), int((y1 - y0) * d)
    def on(img, rect):
        c = np.zeros((h, w), np.float32); ox, oy = int((rect[0] - x0) * d), int((rect[1] - y0) * d)
        c[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img[..., 3]; return c
    U, L = on(up, r_up) > 0.5, on(low, r_low) > 0.5
    ir_ = meta[f"iris_{side}"]["rect"]
    cols = np.nonzero(U.any(0))[0]
    # the opening: about a third of an iris of white either side, never wider than the lash itself
    xa = max(cols.min(), int((ir_[0] - 0.36 * ir_[2] - x0) * d)); xb = min(cols.max(), int((ir_[0] + 1.36 * ir_[2] - x0) * d))
    mid = int((ir_[0] + ir_[2] / 2 - x0) * d); band = range(mid - int(0.2 * ir_[2] * d), mid + int(0.2 * ir_[2] * d))
    def run_bottom(col):   # the lowest solid run of the upper lash in this column: (its top, its bottom)
        ys = np.nonzero(col)[0]
        if not len(ys): return None
        k = len(ys) - 1
        while k > 0 and ys[k - 1] == ys[k] - 1: k -= 1
        return ys[k], ys[-1]
    runs = [run_bottom(U[:, x]) for x in band]; runs = [r for r in runs if r]
    ya = int(np.median([(a_ + b_) / 2 for a_, b_ in runs]))                   # tucked half-way into the lash stroke
    lows = [np.nonzero(L[:, x])[0] for x in band]; lows = [v for v in lows if len(v)]
    yb = int(np.median([(v[0] + v[-1]) / 2 for v in lows])) + 1               # down to the middle of the lower line
    ew = src[f"eyewhite_{side}"]
    ex0, ey0, ex1, ey1 = reg.bbox(ew[..., 3], 0.3)
    fit = reg._resize(ew[ey0:ey1, ex0:ex1], xb - xa, yb - ya)
    img = np.zeros((h, w, 4), np.float32); img[ya:yb, xa:xb] = fit
    # no skin may show between the white and the upper lash: fill up to the lash wherever the shape falls short
    al = img[..., 3]
    for x in range(xa, xb):
        r_ = run_bottom(U[:, x]); ys = np.nonzero(al[:, x] > 0.5)[0]
        if r_ and len(ys) and ys[0] > r_[1]:
            img[r_[1]:ys[0], x, :3] = img[ys[0], x, :3]; img[r_[1]:ys[0], x, 3] = 1.0
    img[..., 3] = np.clip(reg.blur(img[..., 3], 1), 0, 1)
    emit(f"eyewhite_{side}", img, (x0, y0, x1 - x0, y1 - y0), d)
    # the lid curves, in base pixels, for the blink rig
    sel = np.linspace(xa + 1, xb - 2, 41).astype(int)
    solid = img[..., 3] > 0.5
    top = np.array([np.nonzero(solid[:, x])[0][0] if solid[:, x].any() else (ya + yb) / 2 for x in sel], np.float32)
    bot = np.array([np.nonzero(solid[:, x])[0][-1] if solid[:, x].any() else (ya + yb) / 2 for x in sel], np.float32)
    marks["eyes"][side] = {"x": (x0 + sel / d).tolist(), "top": (y0 + top / d).tolist(), "bot": (y0 + bot / d).tolist()}

# Mouth and nose: where the base draws them, and in what colours.
r, g, b, a = (base[..., i] for i in range(4))
il, ir = meta["iris_l"]["rect"], meta["iris_r"]["rect"]
ex, ey, ed = (il[0] + il[2] / 2 + ir[0] + ir[2] / 2) / 2, (il[1] + il[3] / 2 + ir[1] + ir[3] / 2) / 2, (ir[0] + ir[2] / 2) - (il[0] + il[2] / 2)
chin = meta["head"]["rect"][1] + meta["head"]["rect"][3]
wx0, wx1, wy0, wy1 = int(ex - 0.42 * ed), int(ex + 0.42 * ed), int(max(il[1] + il[3], ir[1] + ir[3]) + 0.04 * ed), int(chin - 0.12 * ed)
win = base[wy0:wy1, wx0:wx1]
skin = np.median(win[..., :3].reshape(-1, 3), 0)
dark = (skin.mean() - win[..., :3].mean(-1) > 0.05) & (win[..., 0] - win[..., 2] > 0.04)
lab, n = layers.components(dark)
best = None
for i in range(1, n + 1):
    my, mx = np.nonzero(lab == i)
    w_, cx_, cy_ = mx.max() - mx.min() + 1, wx0 + (mx.min() + mx.max()) / 2, wy0 + np.median(my)
    # a mouth is a centred mark of modest width; the lowest such one wins over a nose drawn above it
    if len(my) < 12 or w_ > 0.6 * (wx1 - wx0) or abs(cx_ - ex) > 0.12 * ed: continue
    if best is None or cy_ > best[1]: best = (cx_, cy_, w_, win[my, mx, :3])
assert best, "no mouth found under the eyes"
marks["mouth"] = {"cx": float(best[0]), "cy": float(best[1]), "w": float(best[2]),
                  "line": np.percentile(best[3], 25, 0).tolist(), "skin": skin.tolist()}
marks["face"] = {"ex": float(ex), "ey": float(ey), "ed": float(ed), "chin": float(chin)}
if os.path.exists(f"{ch}/face_hd.png"):
    # a drawn nose: lift it out of the sharp face redraw by how far each texel sits from the skin around it
    box = json.load(open(f"{ch}/face_crop.json"))["box"]
    hd = np.asarray(Image.open(f"{ch}/face_hd.png").convert("RGB")).astype(np.float32) / 255
    k = (box[2] - box[0]) / hd.shape[1]
    top_y = marks["mouth"]["cy"] - 0.16 * ed
    nx0, nx1, ny0, ny1 = [int(v) for v in ((ex - 0.2 * ed - box[0]) / k, (ex + 0.2 * ed - box[0]) / k, (ey + 0.12 * ed - box[1]) / k, (top_y - box[1]) / k)]
    patch = hd[ny0:ny1, nx0:nx1]
    local = np.stack([reg.blur(patch[..., c], 0.06 * patch.shape[1]) for c in range(3)], -1)
    dist = np.abs(patch - local).sum(-1)
    alpha = np.clip((dist - 0.05) / 0.12, 0, 1)
    fade = np.minimum.outer(np.minimum(np.arange(patch.shape[0]), np.arange(patch.shape[0])[::-1]), np.minimum(np.arange(patch.shape[1]), np.arange(patch.shape[1])[::-1]))
    alpha *= np.clip(fade / (0.12 * patch.shape[1]), 0, 1)
    nose = np.dstack([patch, alpha]).astype(np.float32)
    nose = reg._resize(nose, nose.shape[1] // 2, nose.shape[0] // 2)
    emit("nose", nose, (box[0] + nx0 * k, box[1] + ny0 * k, (nx1 - nx0) * k, (ny1 - ny0) * k), 0.5 / k)

json.dump({"canvas": [W, H], "parts": meta, "marks": marks}, open(f"{out}/parts.json", "w"), indent=1)

# preview: rebuild the picture from the exported parts
ORDER = ["backhair", "tail", "bodyna", "arms_l", "arms_r", "fins_l", "fins_r", "head", "eyewhite_l", "eyewhite_r", "iris_l", "iris_r",
         "nose", "lashlow_l", "lashlow_r", "lashup_l", "lashup_r", "brows_l", "brows_r", "fronthair", "bow_l", "bow_r"]
comp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for name in ORDER:
    if name not in meta: continue
    m = meta[name]; im = Image.open(f"{out}/{name}.png")
    x, y, w, h = m["rect"]
    if name.startswith("iris"):
        # clip to the eye white, as the model's mask will
        ew = meta["eyewhite_" + name[-1]]; em = Image.open(f"{out}/eyewhite_{name[-1]}.png").getchannel("A")
        canvas = Image.new("L", (W, H), 0); canvas.paste(em.resize((int(ew["rect"][2]), int(ew["rect"][3])), Image.LANCZOS), (int(ew["rect"][0]), int(ew["rect"][1])))
        im = im.resize((int(w), int(h)), Image.LANCZOS)
        a = np.minimum(np.asarray(im.getchannel("A")), np.asarray(canvas.crop((int(x), int(y), int(x) + im.width, int(y) + im.height))))
        im.putalpha(Image.fromarray(a))
    else:
        im = im.resize((int(w), int(h)), Image.LANCZOS)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); layer.paste(im, (int(x), int(y))); comp = Image.alpha_composite(comp, layer)
comp.save(f"{ch}/parts_composite.png")
hx, hy, hw, hh = meta["head"]["rect"]; m = 0.2 * hw
boxes = {"head": (hx - m, hy - m, hx + hw + m, hy + hh + m), "eyes": (hx + 0.05 * hw, hy + 0.3 * hh, hx + 0.95 * hw, hy + 0.85 * hh), "full": (0, 0, W, H)}
for k, bx in boxes.items():
    bx = tuple(int(v) for v in bx)
    b = Image.new("RGBA", (W, H), (255, 0, 255, 255) if k != "eyes" else (150, 230, 150, 255))
    pair = []
    for im in (Image.open(f"{ch}/base.png").convert("RGBA"), comp):
        c = b.copy(); c.alpha_composite(im); pair.append(c.crop(bx).convert("RGB"))
    wv = 800; hv = int(wv * pair[0].height / pair[0].width)
    o = Image.new("RGB", (wv * 2, hv)); o.paste(pair[0].resize((wv, hv), Image.LANCZOS), (0, 0)); o.paste(pair[1].resize((wv, hv), Image.LANCZOS), (wv, 0))
    o.save(f"prev/{ch}_parts_{k}.jpg", quality=92)
print({k: os.path.getsize(f"{out}/{v['file']}") // 1024 for k, v in meta.items()}, "total KB", sum(os.path.getsize(f"{out}/{v['file']}") for v in meta.values()) // 1024)
