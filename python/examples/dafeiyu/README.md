# 大肥鱼: from generated layers to a rigged model

Two rigs of one character, a chibi (`q`) and a grown-up (`adult`), built by
script from image-model output. They are the models behind the site's
interactive page (`apps/site/models/dafeiyu-*.clm`).

`rig.py` is the part to read if you want to rig your own art: it turns placed
part textures into a complete model structure and imports it in one
`client.from_json` call. The steps before it are specific to art that arrives
as separately generated layers.

```sh
cargo build --release -p catchlight-editor-server -p catchlight-cli
python3 python/examples/dafeiyu/rig.py q        # writes apps/site/models/dafeiyu-q.clm
python3 python/examples/dafeiyu/rig.py adult
python3 python/examples/dafeiyu/shots.py q turn head rest head_yaw=30 head_roll=-30 eye_left_open=0
```

`rig.py` needs Pillow and numpy and nothing else. `shots.py` renders poses with
`catchlight-cli` into a contact sheet under `workspace/dafeiyu/render/`, which
is how every deform here was checked.

## The pipeline

| Step | Script | In | Out |
| --- | --- | --- | --- |
| 1. Draw | an image model, `prompts.json` | a character sheet | `base.png`, a front-facing illustration |
| 2. Separate | the same model, one request per layer | `base.png` | `L_*.png` body layers, `F_*.png` eye layers |
| 3. Register | `align.py` | the layers | `transforms.json`: each layer's place on the base |
| 4. Cut | `parts.py` | layers and transforms | `parts/<ch>/*.png`, `parts.json` |
| 5. Rig | `rig.py` | `parts/<ch>/` | `dafeiyu-<ch>.clm` |

Steps 1 to 4 ran once; their output, `parts/`, is committed, and the raw
layers (a few hundred megabytes) are not. `align.py` and `parts.py` are kept
as the record of how the parts were made, and run from a directory holding
`<ch>/base.png` and the generated layers.

**Separating.** A model asked for "only the back hair, on the same canvas"
understands the request and redraws the layer complete, hidden parts included,
but at its own scale and position. `prompts.json` holds what was asked.
Asking for an edit of an already separated layer ("this body, arms removed")
keeps proportions far better than asking for the layer from the base again.

**Registering** (`reg.py`). Each layer is placed back on the base by a
similarity transform: masked zero-mean cross-correlation over every scale and
translation, then a refinement on an occlusion-tolerant colour cost. Layers
the base mostly hides are placed by what they must cover instead: the back
hair by overlap with hair-coloured pixels, the head by covering the face skin
without spilling past the jaw, the lower lash line and eye white from the iris
and upper lash.

**Cutting.** Parts are resampled into their own rectangles, small face parts
at a higher texel density. Edge colours are bled under transparent texels.
The eye white is the generated shape stretched into the opening the lashes
leave, and the lid curves the blink needs are read back off it.

## What `rig.py` builds

The world is the base illustration scaled so the figure is about a thousand
units tall: a mesh group indexes its area one cell per world unit.

- **Parts are grids** over their art that skip empty cells. Two parts may
  share a texture and split its cells (the ahoge is cut from the front hair
  this way, so it can swing on its own lattice).
- **Motion is lattices.** Every moving thing sits under a mesh group whose
  grid is deformed by a function of position: a head turn slides each layer
  by its depth over the head, a tilt is a rotation about the neck, the body
  leans from the ankles. Functions are sampled at each binding's keys.
- **A part is carried only by the lattice directly above it,** and a nested
  lattice's own vertices by the one above that. So every meshed node must lie
  wholly inside its parent lattice; the script checks this and prints any
  vertex left outside. A vertex outside is simply not moved, which shows up
  as a stretched sliver.
- **Eyes** are an eye white, an iris masked by it, and upper and lower lashes.
  One two-param binding (open, smile) moves all of them along the measured
  lid curves, so a blink and a smiling squint are the same rig.
- **The mouth** is one drawn part whose mesh is its two lip curves; closed,
  the strokes meet and read as a line.
- **Springs.** Motion responses follow mixes of the posed params and write
  their lag to sway params bound to the hair, ahoge, fins, tail and skirt.

Param names follow `tests/models/lan.clm` where they mean the same thing
(`head_yaw`, `eye_left_open`, `mouth_open`, `body_roll`, ...), so the site's
face tracking drives these rigs unchanged. "Left" is the character's left.
