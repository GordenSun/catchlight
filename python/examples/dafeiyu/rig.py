"""rig.py <q|adult>: placed part textures -> a rigged Catchlight model (.clm).

World space is the base illustration's pixels, Y up, origin on the floor under the face.
Everything that moves is a mesh-group lattice or a per-vertex deform, computed here from
a few measurements of the art, so the same script rigs both characters."""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
import catchlight

ch = sys.argv[1]
PD = f"{HERE}/parts/{ch}"
spec = json.load(open(f"{PD}/parts.json"))
P, M = spec["parts"], spec["marks"]
# The model goes where the site bundles it from; drawn textures and the render-check sidecar stay out of git.
OUT = os.path.join(REPO, "apps", "site", "models")
SCRATCH = os.path.join(REPO, "workspace", "dafeiyu"); os.makedirs(f"{SCRATCH}/out", exist_ok=True)
SYN = f"{SCRATCH}/syn/{ch}"; os.makedirs(SYN, exist_ok=True)

CX = M["face"]["ex"]
FLOOR = P["bodyna"]["rect"][1] + P["bodyna"]["rect"][3]
# Mesh groups index their area one cell per world unit, so the world is kept near a thousand units tall.
S = 0.4
def wx(x): return (np.asarray(x, np.float64) - CX) * S
def wy(y): return (FLOOR - np.asarray(y, np.float64)) * S
def wrect(r):
    """base-pixel rect (x, y, w, h) -> world (x0, y0, x1, y1), y0 the bottom"""
    return float(wx(r[0])), float(wy(r[1] + r[3])), float(wx(r[0] + r[2])), float(wy(r[1]))

nodes, bindings, params, textures = [], [], [], {}
IDENT = {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]}
verts_of = {}


def node(id, parent, kind, name=None, z=0.0):
    nodes.append({"id": id, "parent": parent, "name": name or id, "enabled": True, "z_order": float(z),
                  "transform": IDENT, "lock_to_root": False, "kind": kind})
    return id


def mesh_json(v, uv, tris):
    return {"verts": [round(float(a), 3) for a in v.ravel()], "uvs": [round(float(a), 5) for a in uv.ravel()],
            "indices": {"U16": [int(i) for i in tris.ravel()]}, "origin": [0.0, 0.0]}


def grid(x0, y0, x1, y1, nx, ny, keep=None):
    """A grid of nx by ny cells over a world rect; `keep[j, i]` drops empty cells (row 0 at the top)."""
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y1, y0, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    v = np.stack([gx.ravel(), gy.ravel()], 1)
    uv = np.stack([(gx.ravel() - x0) / (x1 - x0), (y1 - gy.ravel()) / (y1 - y0)], 1)
    tris = []
    for j in range(ny):
        for i in range(nx):
            if keep is not None and not keep[j, i]: continue
            a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i, (j + 1) * (nx + 1) + i + 1
            tris += [[a, b, d], [a, d, c]]
    tris = np.array(tris)
    used = np.unique(tris); remap = -np.ones(len(v), int); remap[used] = np.arange(len(used))
    return v[used], uv[used], remap[tris]


def part(id, parent, tex, z, cell, name=None, opacity=1.0, masks=(), rect=None, file=None, rows=None):
    """A textured part meshed as a grid that skips cells its alpha never touches.

    `rows(cx, cy)` keeps only the cells whose centres it accepts, so two parts can share one
    texture and split it between them."""
    r = wrect(P[tex]["rect"]) if rect is None else rect
    path = file or f"{PD}/{P[tex]['file']}"
    alpha = Image.open(path).getchannel("A")
    nx, ny = max(2, math.ceil((r[2] - r[0]) / cell)), max(2, math.ceil((r[3] - r[1]) / cell))
    keep = np.asarray(alpha.resize((nx, ny), Image.BOX)) > 0
    if rows is not None:
        cx = r[0] + (np.arange(nx) + 0.5) * (r[2] - r[0]) / nx
        cy = r[3] - (np.arange(ny) + 0.5) * (r[3] - r[1]) / ny
        keep &= rows(cx[None, :], cy[:, None])
    v, uv, tris = grid(*r, nx, ny, keep)
    tid = f"tex-{tex or id}"
    textures[tid] = path
    node(id, parent, {"Part": {"mesh": mesh_json(v, uv, tris), "albedo": tid, "opacity": opacity, "blend_mode": "Normal",
                               "tint": [1.0, 1.0, 1.0], "screen_tint": [0.0, 0.0, 0.0],
                               "masks": [{"source": m, "mode": "Mask"} for m in masks], "mask_threshold": 0.5, "slots": []}}, name, z)
    verts_of[id] = v
    return v


def lattice(id, parent, rect, nx, ny, name=None):
    v, uv, tris = grid(*rect, nx, ny)
    node(id, parent, {"MeshGroup": {"mesh": mesh_json(v, uv, tris), "translate_children": True}}, name)
    verts_of[id] = v
    return v


def param(id, lo, hi, default=0.0):
    params.append({"id": id, "name": id, "min": float(lo), "max": float(hi), "default": float(default)})
    return id


def bind(nid, pids, keys, fn):
    """A deform binding: `fn(verts, *values)` gives each vertex's offset at every key combination."""
    v = verts_of[nid]
    pinfo = [next(p for p in params if p["id"] == pid) for pid in pids]
    pos = [[(k - p["min"]) / (p["max"] - p["min"]) for k in ks] for ks, p in zip(keys, pinfo)]
    cells = []
    for j, ky in enumerate(keys[1] if len(keys) > 1 else [None]):
        for i, kx in enumerate(keys[0]):
            off = np.asarray(fn(v, kx) if ky is None else fn(v, kx, ky), np.float64)
            cells.append({"x": i, "y": j, "value": [round(float(a), 3) for a in off.ravel()]})
    bindings.append({"params": list(pids), "key_positions": pos, "node": nid, "interpolate_mode": "Linear", "values": {"Deform": {"cells": cells}}})


def bind_scalar(nid, pid, target, pairs):
    p = next(p for p in params if p["id"] == pid)
    bindings.append({"params": [pid], "key_positions": [[(k - p["min"]) / (p["max"] - p["min"]) for k, _ in pairs]], "node": nid,
                     "interpolate_mode": "Linear", "values": {target: {"cells": [{"x": i, "y": 0, "value": float(val)} for i, (_, val) in enumerate(pairs)]}}})


def response(id, out, inputs, period, damping, parent="root"):
    node(id, parent, {"MotionResponse": {"inputs": [{"source": {"kind": "pose_param", "param": p} if p != "parent" else {"kind": "parent"}, "gain": g} for p, g in inputs],
                                        "period_s": period, "damping_ratio": damping, "output_param": out}})
    return id


def rot(v, pivot, ang):
    """Offsets that turn each vertex about `pivot` by its own angle (radians, counterclockwise)."""
    ang = np.broadcast_to(np.asarray(ang, np.float64), (len(v),))
    d = v - pivot
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1) - d


def ramp(t): return np.clip(t, 0.0, 1.0)
def smooth(t): t = ramp(t); return t * t * (3 - 2 * t)
def grow(r, m): return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)
def union(*rs): return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))

# ---- measurements ---------------------------------------------------------------------------
R_ = {k: wrect(v["rect"]) for k, v in P.items()}
head = R_["head"]
HR = (head[2] - head[0]) / 2                      # head radius: the unit most amounts are given in
HC = np.array([0.0, (head[1] + head[3]) / 2])     # head centre
CHIN = head[1]
NECK = np.array([0.0, CHIN + 0.12 * HR])          # where the head tilts about
TOP = max(r[3] for r in R_.values())
EVERY = grow(union(*R_.values()), 0.6 * HR)
ED = M["face"]["ed"] * S
EYE_Y = float(wy(M["face"]["ey"]))
body = R_["bodyna"]
# the hem is the lowest wide row of the body art; legs are the narrow rows under it
ba = np.asarray(Image.open(f"{PD}/bodyna.png").getchannel("A")) > 40
roww = ba.sum(1); wide = np.nonzero(roww > 0.6 * roww.max())[0]
HEM = body[3] - (wide.max() + 1) / ba.shape[0] * (body[3] - body[1])
WAIST = HEM + {"q": 0.5, "adult": 0.62}[ch] * (body[3] - HEM)
HIP = {"q": 0.42, "adult": 0.5}[ch] * body[3]
ANKLE = 0.07 * body[3]
SHOULDER = body[3] - 0.12 * (body[3] - WAIST)
fh, bh = R_["fronthair"], R_["backhair"]
# the ahoge is the narrow run of rows at the top of the front-hair art
fa = np.asarray(Image.open(f"{PD}/fronthair.png").getchannel("A")) > 40
fw = fa.sum(1); first = np.nonzero(fw > 0)[0][0]
band = np.nonzero(fw > 0.5 * fw.max())[0][0]
AHOGE_Y = fh[3] - band / fa.shape[0] * (fh[3] - fh[1])
acols = np.nonzero(fa[first:band].any(0))[0]
AHOGE_X = fh[0] + (acols.min() + acols.max()) / 2 / fa.shape[1] * (fh[2] - fh[0])
print(f"{ch}: head r {HR:.0f} centre y {HC[1]:.0f} chin {CHIN:.0f} top {TOP:.0f} hem {HEM:.0f} waist {WAIST:.0f} ahoge base {AHOGE_Y:.0f} at x {AHOGE_X:.0f}")

# ---- params ----------------------------------------------------------------------------------
for n in ["head_yaw", "head_pitch", "head_roll"]: param(n, -30, 30)
param("body_yaw", -10, 10); param("body_roll", -10, 10); param("body_move_x", -10, 10); param("breath", 0, 1)
for s in ["left", "right"]:
    param(f"eye_{s}_open", 0, 1, 1); param(f"eye_{s}_smile", 0, 1)
    param(f"eye_{s}_gaze_x", -1, 1); param(f"eye_{s}_gaze_y", -1, 1)
    param(f"eyebrow_{s}_height", -1, 1); param(f"eyebrow_{s}_inner_height", -1, 1)
    param(f"arm_{s}_raise", 0, 1)
param("mouth_open", 0, 1); param("mouth_expression", -1, 1); param("face_blush", 0, 1)
param("fin_flap", -1, 1); param("tail_wag", -1, 1)
for n in ["hair_front_sway", "hair_back_sway", "hair_back_sway_tip", "ahoge_sway", "fin_sway", "tail_sway", "skirt_sway"]: param(n, -1, 1)
A3 = [-30, 0, 30]
A5 = [-30, -15, 0, 15, 30]
S3 = [-1, 0, 1]
rad = math.radians

# ---- tree ------------------------------------------------------------------------------------
node("root", None, "Group", "大肥鱼" + {"q": " Q版", "adult": " 成年版"}[ch])
# A lattice only carries what lies inside it, its child lattices' own vertices included,
# so each one is kept strictly inside its parent with room for what it holds to move.
UNIT = 0.1 * TOP
lattice("mg-body", "root", grow(EVERY, 4 * UNIT), 16, 20, "Body")
lattice("mg-headroll", "mg-body", grow(EVERY, 2 * UNIT), 10, 12, "Head Roll")

# depth of each layer over the head, in head radii: what a turn slides it by
def depth(v, z0, z1, rho, cy=0.0):
    u, w = (v[:, 0] - HC[0]) / HR, (v[:, 1] - HC[1]) / HR - cy
    return z0 + z1 * np.clip(1 - (u * u + w * w) / (rho * rho), 0, 1)
def turn(zf):
    def f(v, yaw, pitch):
        z = zf(v)
        return np.stack([HR * math.sin(rad(yaw)) * z, HR * math.sin(rad(pitch)) * z], 1)
    return f
def hang(lo, wmin):
    """Counter-turn for hair below the neck, so a head tilt leaves long hair hanging rather than swinging rigid."""
    def f(v, roll):
        w = 1 - (1 - wmin) * smooth((NECK[1] - v[:, 1]) / max(NECK[1] - lo, 1))
        return rot(v, NECK, rad(roll) * (w - 1))
    return f
def sway(top, bottom, amp, power=1.5, start=0.0):
    def f(v, s):
        t = ramp(((top - v[:, 1]) / (top - bottom) - start) / (1 - start)) ** power
        return np.stack([s * amp * t, -abs(s) * amp * 0.12 * t * t], 1)
    return f

# back hair
lattice("mg-backhair", "mg-headroll", grow(bh, 0.25 * HR), 12, 14, "Back Hair")
part("backhair", "mg-backhair", "backhair", -60, 0.16 * HR, "back hair")
bind("mg-backhair", ["head_yaw", "head_pitch"], [A3, A3], turn(lambda v: depth(v, 0.0, -0.32, 2.2)))
bind("mg-backhair", ["head_roll"], [A5], hang(bh[1], 0.12))
bind("mg-backhair", ["hair_back_sway"], [S3], sway(HC[1], bh[1], 0.75 * HR))
bind("mg-backhair", ["hair_back_sway_tip"], [S3], sway(HC[1], bh[1], 0.6 * HR, 1.6, 0.45))

# tail: curls from its root behind the skirt
tl = R_["tail"]
TAIL_ROOT = np.array([tl[0] + 0.12 * (tl[2] - tl[0]), tl[1] + 0.3 * (tl[3] - tl[1])])
TAIL_LEN = math.hypot(tl[2] - TAIL_ROOT[0], tl[3] - TAIL_ROOT[1])
def curl(root, length, deg, power=1.0):
    def f(v, s):
        d = np.hypot(v[:, 0] - root[0], v[:, 1] - root[1]) / length
        return rot(v, root, rad(deg) * s * ramp(d) ** power)
    return f
lattice("mg-tail", "mg-body", grow(tl, 0.3 * HR), 8, 8, "Tail")
part("tail", "mg-tail", "tail", -55, 0.14 * HR)
bind("mg-tail", ["tail_wag"], [[-1, -0.5, 0, 0.5, 1]], curl(TAIL_ROOT, TAIL_LEN, 22))
bind("mg-tail", ["tail_sway"], [S3], curl(TAIL_ROOT, TAIL_LEN, 18, 2.2))

# torso
lattice("mg-torso", "mg-body", grow(body, 0.15 * HR), 10, 14, "Torso")
part("body", "mg-torso", "bodyna", -50, 0.14 * HR)
def skirt(v, s):
    t = ramp((WAIST - v[:, 1]) / (WAIST - HEM)) ** 1.5 * (1 - smooth((HEM - v[:, 1]) / (0.08 * (WAIST - HEM)) - 0.2))
    return np.stack([s * 0.2 * HR * t, np.zeros(len(v))], 1)
bind("mg-torso", ["skirt_sway"], [S3], skirt)

# arms turn about the shoulder
for side, char in (("l", "right"), ("r", "left")):
    ar = R_[f"arms_{side}"]; sg = 1 if side == "l" else -1
    pivot = np.array([(ar[2] if side == "l" else ar[0]) - sg * 0.14 * (ar[2] - ar[0]), ar[3] - 0.14 * (ar[3] - ar[1])])
    lattice(f"mg-arm-{side}", "mg-body", grow(ar, 0.2 * HR), 6, 6, f"Arm {char}")
    part(f"arm-{side}", f"mg-arm-{side}", f"arms_{side}", {"q": -45, "adult": -47}[ch], 0.12 * HR)
    top = {"q": 100, "adult": 118}[ch]
    bind(f"mg-arm-{side}", [f"arm_{char}_raise"], [[0, 0.25, 0.5, 0.75, 1]], lambda v, s, p=pivot, g=sg, t=top: rot(v, p, -g * rad(t) * s))

# fins and bows
for side in "lr":
    fr = R_[f"fins_{side}"]; sg = 1 if side == "l" else -1
    root = np.array([fr[2] if side == "l" else fr[0], (fr[1] + fr[3]) / 2 + 0.1 * (fr[3] - fr[1])])
    lattice(f"mg-fin-{side}", "mg-headroll", grow(fr, 0.2 * HR), 6, 5, f"Fin {side}")
    part(f"fin-{side}", f"mg-fin-{side}", f"fins_{side}", -44, 0.1 * HR)
    L = fr[2] - fr[0]
    bind(f"mg-fin-{side}", ["head_yaw", "head_pitch"], [A3, A3], turn(lambda v: depth(v, 0.06, 0.0, 1.0)))
    bind(f"mg-fin-{side}", ["fin_flap"], [S3], lambda v, s, r=root, g=sg, L=L: rot(v, r, -g * rad(16) * s * ramp(np.abs(v[:, 0] - r[0]) / L) ** 0.8))
    bind(f"mg-fin-{side}", ["fin_sway"], [S3], lambda v, s, r=root, L=L: rot(v, r, rad(11) * s * ramp(np.abs(v[:, 0] - r[0]) / L) ** 1.3))

# face
face_rect = grow(union(head, *[v for k, v in R_.items() if k.startswith(("lash", "brows", "iris", "eyewhite", "nose"))]), 0.4 * HR)
lattice("mg-face", "mg-headroll", face_rect, 12, 12, "Face")
face_z = lambda v: depth(v, 0.04, 0.6, 1.1)
bind("mg-face", ["head_yaw", "head_pitch"], [A3, A3], turn(face_z))
part("face", "mg-face", "head", -40, 0.1 * HR)
if "nose" in P: part("nose", "mg-face", "nose", -38.5, 0.05 * HR)

# blush: two soft patches with a few hatch strokes, faded in by face_blush
def blush_tex():
    w, h = 640, 220
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow = Image.new("L", (w, h), 0); d = ImageDraw.Draw(glow)
    for cx in (130, 510): d.ellipse((cx - 105, 40, cx + 105, 180), fill=150)
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    im = Image.merge("RGBA", [Image.new("L", (w, h), c) for c in (255, 120, 130)] + [glow])
    hatch = Image.new("L", (w, h), 0); d = ImageDraw.Draw(hatch)
    for cx in (130, 510):
        for k in (-34, 0, 34): d.line((cx + k + 14, 78, cx + k - 14, 142), fill=170, width=7)
    hatch = hatch.filter(ImageFilter.GaussianBlur(1.6))
    im.alpha_composite(Image.merge("RGBA", [Image.new("L", (w, h), c) for c in (240, 90, 110)] + [hatch]))
    im.save(f"{SYN}/blush.png"); return f"{SYN}/blush.png"
bw = {"q": 1.9, "adult": 1.6}[ch] * ED; by = EYE_Y - {"q": 0.36, "adult": 0.42}[ch] * ED
part("blush", "mg-face", None, -38, 0.12 * HR, rect=(-bw / 2, by - bw * 0.17, bw / 2, by + bw * 0.17), file=blush_tex())
bind_scalar("blush", "face_blush", "Opacity", [(0, 0.0), (1, 1.0)])

# eyes: the lid curves measured from the lash art drive the blink
for side, char in (("l", "right"), ("r", "left")):
    e = M["eyes"][side]
    ex_, top_, bot_ = wx(e["x"]), wy(e["top"]), wy(e["bot"])
    xc, half = (ex_[0] + ex_[-1]) / 2, (ex_[-1] - ex_[0]) / 2
    eh = float(np.max(top_ - bot_))
    cell = (ex_[-1] - ex_[0]) / 16
    def lids(x, o, s, ex_=ex_, top_=top_, bot_=bot_, xc=xc, half=half, eh=eh):
        """upper and lower lid heights at x for an eye `o` open and `s` smiling"""
        t, b = np.interp(x, ex_, top_), np.interp(x, ex_, bot_)
        low = b + s * 0.42 * eh * np.clip(1 - ((x - xc) / half) ** 2, 0, 1)
        return np.maximum(low + (t - low) * o, low), low, t, b
    part(f"eyewhite-{side}", "mg-face", f"eyewhite_{side}", -37, cell)
    part(f"iris-{side}", "mg-face", f"iris_{side}", -36, cell * 1.5, masks=[f"eyewhite-{side}"])
    part(f"lashlow-{side}", "mg-face", f"lashlow_{side}", -34, cell)
    part(f"lashup-{side}", "mg-face", f"lashup_{side}", -33, cell)
    def white(v, o, s):
        up, low, t, b = lids(v[:, 0], o, s)
        f = np.clip((t - v[:, 1]) / np.maximum(t - b, 1e-3), -0.3, 1.3)
        return np.stack([np.zeros(len(v)), up + (low - up) * f - v[:, 1]], 1)
    def upper(v, o, s):
        up, low, t, b = lids(v[:, 0], o, s)
        return np.stack([np.zeros(len(v)), up - t], 1)
    def lower(v, o, s):
        up, low, t, b = lids(v[:, 0], o, s)
        return np.stack([np.zeros(len(v)), low - b], 1)
    keys = [[0, 0.5, 1], [0, 1]]
    for nid, fn in ((f"eyewhite-{side}", white), (f"lashup-{side}", upper), (f"lashlow-{side}", lower)):
        bind(nid, [f"eye_{char}_open", f"eye_{char}_smile"], keys, fn)
    ir = R_[f"iris_{side}"]
    bind(f"iris-{side}", [f"eye_{char}_gaze_x", f"eye_{char}_gaze_y"], [S3, S3],
         lambda v, gx, gy, w=ir[2] - ir[0], h=ir[3] - ir[1]: np.tile([gx * 0.17 * w, gy * 0.12 * h], (len(v), 1)))
    # brows ride above the bangs, a little see-through, as the base draws them
    br = R_[f"brows_{side}"]; bc = np.array([(br[0] + br[2]) / 2, (br[1] + br[3]) / 2]); sg = 1 if side == "l" else -1
    part(f"brow-{side}", "mg-face", f"brows_{side}", -8, (br[2] - br[0]) / 10, opacity=0.85)
    bind(f"brow-{side}", [f"eyebrow_{char}_height"], [S3], lambda v, s: np.tile([0.0, s * 0.13 * ED], (len(v), 1)))
    bind(f"brow-{side}", [f"eyebrow_{char}_inner_height"], [S3], lambda v, s, c=bc, g=sg: rot(v, c, g * rad(16) * s))

# mouth: one lens-shaped part drawn in its own (u, v) space; its two lip curves are the rig
mo = M["mouth"]
MX, MY, MW = float(wx(mo["cx"])), float(wy(mo["cy"])), mo["w"] * S * {"q": 1.0, "adult": 1.15}[ch]
line = tuple(int(255 * min(c, d)) for c, d in zip(mo["line"], (0.55, 0.26, 0.25)))
def mouth_tex():
    w, h = 512, 400
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); px = im.load()
    for y in range(h):
        v = y / (h - 1)
        for x in range(w):
            u = x / (w - 1) * 2 - 1
            if v < 0.1 or v > 0.9: continue
            if v < 0.2 or v > 0.8: px[x, y] = line + (255,); continue
            t = (v - 0.2) / 0.6
            col = [int(a + (b - a) * t) for a, b in zip((112, 36, 48), (176, 62, 74))]
            # tongue: a mound rising from the lower lip
            if (u / 0.62) ** 2 + ((1 - t) / 0.5) ** 2 < 1: col = [240, 128, 136] if (u / 0.5) ** 2 + ((1 - t - 0.1) / 0.3) ** 2 > 1 else [248, 150, 156]
            if ch == "q" and 0.3 < u < 0.52 and t < 0.34 * (1 - abs(u - 0.41) / 0.11): col = [255, 255, 255]   # the little fang
            if ch == "adult" and abs(u) < 0.5 and t < 0.1: col = [255, 250, 250]                                 # a glimpse of teeth
            px[x, y] = tuple(col) + (255,)
    a = im.getchannel("A").filter(ImageFilter.GaussianBlur(1.2)); im.putalpha(a)
    im.save(f"{SYN}/mouth.png"); return f"{SYN}/mouth.png"
US = np.linspace(-1.12, 1.12, 21)
VS = np.array([0.0, 0.1, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.9, 1.0])
T = {"q": 0.05, "adult": 0.035}[ch] * MW + 1.2 * S     # stroke thickness
def mouth_shape(o, e):
    """world positions of the mouth mesh, `o` open and `e` from frown (-1) to smile (+1)"""
    hw = MW / 2 * (1 + 0.16 * e) * (1 - {"q": 0.12, "adult": 0.3}[ch] * o)
    out = np.zeros((len(VS), len(US), 2))
    for i, u in enumerate(US):
        uc = float(np.clip(u, -1, 1)); edge = math.sqrt(max(0.0, 1 - uc ** 4))
        if ch == "q":   # the cat mouth: a soft "w" whose middle peak flattens as it opens or frowns
            closed = MW * (0.10 * e * uc * uc + 0.085 * max(0.0, 1 - abs(e) * 0.5) * (math.cos(2 * math.pi * uc) - 1) / 2 + 0.07 * uc * uc)
        else:
            closed = MW * (0.05 + 0.13 * e) * uc * uc
        yt = MY + closed * (1 - 0.5 * o) + o * MW * 0.06
        yb = yt - o * MW * {"q": 0.62, "adult": 0.36}[ch] * (1 + 0.15 * e) * edge
        tau = T * (0.35 + 0.65 * edge) * (0.5 + 0.5 * o)
        ti = min(tau / 2, (yt - yb) / 2)
        ys = [yt + tau / 2 + 0.6 * T, yt + tau / 2, yt - ti] + [yt - ti + (yb + ti - (yt - ti)) * k / 5 for k in range(1, 5)] + [yb + ti, yb - tau / 2, yb - tau / 2 - 0.6 * T]
        out[:, i, 0] = MX + hw * u; out[:, i, 1] = ys
    return out.reshape(-1, 2)
rest = mouth_shape(0, 0)
uvm = np.stack(np.meshgrid((US / 1.12 + 1) / 2, VS), -1).reshape(-1, 2)
tris = []
for j in range(len(VS) - 1):
    for i in range(len(US) - 1):
        a, b, c, d = j * len(US) + i, j * len(US) + i + 1, (j + 1) * len(US) + i, (j + 1) * len(US) + i + 1
        tris += [[a, b, d], [a, d, c]]
textures["tex-mouth"] = mouth_tex()
node("mouth", "mg-face", {"Part": {"mesh": mesh_json(rest, uvm, np.array(tris)), "albedo": "tex-mouth", "opacity": 1.0, "blend_mode": "Normal",
                                   "tint": [1.0, 1.0, 1.0], "screen_tint": [0.0, 0.0, 0.0], "masks": [], "mask_threshold": 0.5, "slots": []}}, "mouth", -35)
verts_of["mouth"] = rest
bind("mouth", ["mouth_open", "mouth_expression"], [[0, 0.5, 1], S3], lambda v, o, e: mouth_shape(o, e) - rest)

# front hair, with the ahoge on its own small lattice
fhl = grow(union(fh, *[R_[k] for k in ("bow_l", "bow_r") if k in R_]), 0.25 * HR); fhl = (fhl[0], fhl[1], fhl[2], fh[3] + 0.9 * HR)
lattice("mg-fronthair", "mg-headroll", fhl, 12, 15, "Front Hair")
# a part is only carried by the lattice directly above it, so the ahoge is its own part: the same
# texture, split at the ahoge's base with a sliver of overlap that hides the seam when it swings
AHOGE_W = (acols.max() - acols.min()) / fa.shape[1] * (fh[2] - fh[0]) / 2 + 0.25 * HR
in_ahoge = lambda cx, cy, pad: (cy > AHOGE_Y + pad) & (np.abs(cx - AHOGE_X) < AHOGE_W)
lattice("mg-ahoge", "mg-fronthair", (AHOGE_X - AHOGE_W - 0.3 * HR, AHOGE_Y - 0.4 * HR, AHOGE_X + AHOGE_W + 0.3 * HR, fh[3] + 0.6 * HR), 6, 6, "Ahoge")
part("fronthair", "mg-fronthair", "fronthair", -10, 0.13 * HR, "front hair", rows=lambda cx, cy: ~in_ahoge(cx, cy, 0.07 * HR))
part("ahoge", "mg-ahoge", "fronthair", -10.5, 0.13 * HR, "ahoge", rows=lambda cx, cy: in_ahoge(cx, cy, -0.07 * HR))
hair_z = lambda v: depth(v, 0.10, 0.72, 1.5, 0.15)
bind("mg-fronthair", ["head_yaw", "head_pitch"], [A3, A3], turn(hair_z))
bind("mg-fronthair", ["head_roll"], [A5], hang(fh[1], 0.4))
bind("mg-fronthair", ["hair_front_sway"], [S3], sway(HC[1] + 0.35 * HR, fh[1], {"q": 0.3, "adult": 0.45}[ch] * HR, 1.4))
AH = fh[3] - AHOGE_Y
bind("mg-ahoge", ["ahoge_sway"], [S3], lambda v, s: rot(v, np.array([AHOGE_X, AHOGE_Y]), -rad(24) * s * ramp((v[:, 1] - AHOGE_Y) / AH) ** 0.8))
for side in "lr":
    if f"bow_{side}" in P: part(f"bow-{side}", "mg-fronthair", f"bow_{side}", -9, 0.08 * HR)

# whole-body motion, on the outermost lattice so everything above rides it
def lean(v, roll):     # from the ankles, bending in up to the hips; positive leans toward screen right
    return rot(v, np.array([0.0, ANKLE]), -rad(roll) * smooth((v[:, 1] - ANKLE) / (HIP - ANKLE)))
def twist(v, yaw):
    up = smooth((v[:, 1] - HIP) / (SHOULDER - HIP))
    return np.stack([0.26 * HR * (yaw / 10) * up * (0.45 + 0.55 * np.clip(1 - (v[:, 0] / (1.3 * HR)) ** 2, 0, 1)), np.zeros(len(v))], 1)
def shift(v, m):
    return np.stack([0.4 * HR * (m / 10) * smooth((v[:, 1] - ANKLE) / (SHOULDER - ANKLE)), np.zeros(len(v))], 1)
def breathe(v, b):
    up = smooth((v[:, 1] - HIP) / (SHOULDER - HIP))
    return np.stack([0.012 * HR * b * up * v[:, 0] / HR, 0.035 * HR * b * up], 1)
bind("mg-body", ["body_roll"], [[-10, -5, 0, 5, 10]], lean)
bind("mg-body", ["body_yaw"], [[-10, 0, 10]], twist)
bind("mg-body", ["body_move_x"], [[-10, 0, 10]], shift)
bind("mg-body", ["breath"], [[0, 1]], breathe)
bind("mg-headroll", ["head_roll"], [A5], lambda v, roll: rot(v, NECK, rad(roll)))

# springs: each follows a mix of posed params and writes its lag to a sway param
response("motion/hair_front_sway", "hair_front_sway", [("head_yaw", 0.03), ("head_roll", 0.03), ("body_move_x", 0.06), ("body_roll", -0.05)], 0.42, 0.42)
response("motion/hair_back_sway", "hair_back_sway", [("head_yaw", 0.028), ("head_roll", 0.034), ("body_move_x", 0.08), ("body_roll", -0.07)], 0.75, 0.38)
response("motion/hair_back_sway_tip", "hair_back_sway_tip", [("parent", 1.0)], 0.6, 0.35, parent="motion/hair_back_sway")
response("motion/ahoge_sway", "ahoge_sway", [("head_yaw", 0.04), ("head_roll", 0.05), ("body_move_x", 0.08), ("head_pitch", 0.015)], 0.34, 0.2)
response("motion/fin_sway", "fin_sway", [("head_roll", 0.04), ("head_yaw", 0.02), ("fin_flap", 0.5), ("body_move_x", 0.05)], 0.3, 0.32)
response("motion/tail_sway", "tail_sway", [("tail_wag", 0.9), ("body_roll", -0.08), ("body_move_x", 0.08), ("body_yaw", 0.05)], 0.62, 0.3)
response("motion/skirt_sway", "skirt_sway", [("body_move_x", 0.09), ("body_roll", -0.09), ("body_yaw", 0.06)], 0.58, 0.34)

# every meshed node must sit wholly inside the lattice directly above it, or its stray vertices are left behind
by_id = {n["id"]: n for n in nodes}
for n in nodes:
    if n["id"] not in verts_of or n["parent"] not in verts_of: continue
    pv, cv = verts_of[n["parent"]], verts_of[n["id"]]
    lo, hi = pv.min(0), pv.max(0)
    out = ((cv <= lo) | (cv >= hi)).any(1)
    if out.any(): print(f"!! {n['id']}: {out.sum()} of {len(cv)} vertices outside {n['parent']} {lo.round()} {hi.round()} ; child spans {cv.min(0).round()} {cv.max(0).round()}")

structure = {"nodes": nodes, "params": params, "bindings": bindings, "welds": [], "animations": []}
meta = {"bounds": union(*R_.values()), "head": [float(HC[0]), float(HC[1]), float(HR)], "top": TOP, "eye_y": EYE_Y}
json.dump(meta, open(f"{SCRATCH}/out/{ch}.meta.json", "w"), indent=1)
# `launch()` finds the server on PATH or in target/debug; a release build, when there is one, is preferred.
built = os.path.join(REPO, "target", "release", "catchlight-editor-server")
with catchlight.launch(binary=built if os.path.exists(built) else None) as server:
    client = server.client()
    session = client.from_json(structure, textures, name=f"dafeiyu-{ch}")
    from catchlight.protocol_gen import Check
    print("check:", client.send(Check(session=session)).warnings)
    client.save_to(session, f"{OUT}/dafeiyu-{ch}.clm")
verts = sum(len(v) for k, v in verts_of.items())
print(f"wrote apps/site/models/dafeiyu-{ch}.clm {os.path.getsize(f'{OUT}/dafeiyu-{ch}.clm') / 1e6:.1f} MB; {len(nodes)} nodes, {len(params)} params, {len(bindings)} bindings, {verts} vertices")
