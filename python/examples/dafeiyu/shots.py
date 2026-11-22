"""shots.py <q|adult> <sheet-name> <view> pose ... : render poses with catchlight-cli into one contact sheet.

view: full | head | face | bust ; pose: comma-separated param=value, or 'rest'. Sheets land in workspace/dafeiyu/render/.
Motion responses are off, so a sway param set here shows its shape rather than settling to zero."""
import json, os, subprocess, sys
from PIL import Image, ImageDraw
ch, name, view = sys.argv[1:4]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SCRATCH = f"{REPO}/workspace/dafeiyu"; os.makedirs(f"{SCRATCH}/render", exist_ok=True)
cli = next(p for p in (f"{REPO}/target/release/catchlight-cli", f"{REPO}/target/debug/catchlight-cli") if os.path.exists(p))
m = json.load(open(f"{SCRATCH}/out/{ch}.meta.json"))
b = m["bounds"]; hx, hy, hr = m["head"]
if view == "full": rect, scale = (b[0] - 40, b[1] - 20, b[2] - b[0] + 80, b[3] - b[1] + 60), 0.42
elif view == "head": rect, scale = (hx - 2.3 * hr, hy - 1.7 * hr, 4.6 * hr, 3.6 * hr), 1.1
elif view == "face": rect, scale = (hx - 1.25 * hr, hy - 1.15 * hr, 2.5 * hr, 2.1 * hr), 2.4
else: rect, scale = (hx - 3.2 * hr, hy - 4.2 * hr, 6.4 * hr, 6.6 * hr), 0.75
cells = []
for i, pose in enumerate(sys.argv[4:]):
    out = f"{SCRATCH}/render/{ch}_{name}_{i}.png"
    cmd = [cli, "render", f"{REPO}/apps/site/models/dafeiyu-{ch}.clm", "-o", out, f"--rect={rect[0]},{rect[1]},{rect[2]},{rect[3]}", "--scale", str(scale), "--motion-responses", "off"]
    if pose != "rest":
        for kv in pose.split(","): cmd += ["--set", kv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-600:]); sys.exit(1)
    im = Image.open(out).convert("RGB"); ImageDraw.Draw(im).text((6, 4), pose, fill=(200, 0, 120)); cells.append(im)
cols = min(len(cells), int(os.environ.get("COLS", 4)))
rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
w, h = cells[0].size
sheet = Image.new("RGB", (w * cols, h * len(rows)), "white")
for j, r in enumerate(rows):
    for i, c in enumerate(r): sheet.paste(c, (i * w, j * h))
sheet.save(f"{SCRATCH}/render/{ch}_{name}.jpg", quality=90); print(f"{SCRATCH}/render/{ch}_{name}.jpg", sheet.size)
