/** What makes a rigged character feel alive without a camera: a pose computed
 * every frame from where the visitor's pointer is, a clock, and whatever they
 * last asked for.
 *
 * Pure arithmetic over param names, with no session and no DOM, so the whole
 * performance can be stepped in a test. The host owns time: it calls
 * [`Performer.step`] with a frame's `dt` and applies the values it returns.
 * A param the model does not carry is simply never returned, so one performer
 * drives any rig that follows the naming in `tests/models/lan.clm`.
 *
 * Layers, lowest first, each writing targets the next may override: idle
 * (breath, sway, a slow tail), gaze following, the current expression, the
 * motion clip in flight, touch reactions, blinking. Targets are then eased per
 * param family, because a head that snaps and an eyelid that drifts both read
 * as a machine. Blinks are the exception and are written through unsmoothed.
 */

export interface ParamRange {
  id: string;
  name: string;
  min: number;
  max: number;
  default: number;
}

export type Expression = "neutral" | "happy" | "shy" | "angry" | "surprised" | "sleepy" | "wink" | "smug";
export type Motion = "wave" | "nod" | "shake" | "tail" | "fins" | "talk" | "sway" | "cheer";
/** Where a touch landed, as the host hit-tested it. */
export type Touch = "head" | "face" | "fin" | "tail" | "body" | "hand" | "ahoge" | "skirt";

export interface PerformerOptions {
  /** Injected so a test can make the performance deterministic. */
  random?: () => number;
}

type Pose = Record<string, number>;

const EXPRESSIONS: Record<Expression, Pose> = {
  neutral: {},
  happy: { eye_smile: 1, eye_open: 0, mouth_open: 0.55, mouth_expression: 1, face_blush: 0.35, brow_height: 0.4 },
  shy: { face_blush: 1, mouth_expression: 0.35, eye_open: 0.72, brow_inner: 0.7, brow_height: -0.1, gaze_y: -0.7, gaze_x: 0.5, pitch: -7, roll: 6 },
  angry: { brow_inner: -1, brow_height: -0.55, mouth_expression: -1, eye_open: 0.78, face_blush: 0.25, pitch: -4 },
  surprised: { eye_open: 1, brow_height: 1, mouth_open: 0.75, mouth_expression: -0.25, pitch: 5 },
  sleepy: { eye_open: 0.32, brow_height: -0.35, brow_inner: 0.3, mouth_expression: -0.1, pitch: -11, roll: -5 },
  wink: { wink: 1, eye_smile: 0.6, mouth_open: 0.3, mouth_expression: 1, face_blush: 0.2, roll: -8 },
  smug: { eye_open: 0.6, eye_smile: 0.35, mouth_expression: 0.8, brow_inner: -0.35, brow_height: 0.2, roll: 7, gaze_x: -0.4 },
};

/** Seconds a param family takes to cover most of the way to its target. */
const EASE = { head: 0.11, body: 0.24, gaze: 0.055, face: 0.09, limb: 0.13 } as const;

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
const smooth = (t: number): number => { const c = clamp(t, 0, 1); return c * c * (3 - 2 * c); };

interface Clip { motion: Motion; t: number; length: number; }
interface Reaction { touch: Touch; t: number; length: number; }

export class Performer {
  /** Where to look, each axis in `-1..=1` with +y up; `undefined` hands the gaze back to idle. */
  look: [number, number] | undefined;
  /** A drag in progress, in the same units as `look`: leans and slides the body. */
  drag: [number, number] | undefined;
  /** Being stroked on the head right now. */
  petting = false;
  /** Lip-sync level in `0..=1` from a microphone, when one is driving the mouth. */
  voice: number | undefined;
  followPointer = true;
  autoBlink = true;
  idle = true;

  readonly #ranges = new Map<string, ParamRange>();
  readonly #random: () => number;
  readonly #current = new Map<string, number>();
  #time = 0;
  #expression: Expression = "neutral";
  #expressionLeft = 0;
  #clip: Clip | undefined;
  #reaction: Reaction | undefined;
  #blinkAt = 1.2;
  #blink = -1;
  #again = false;
  #glanceAt = 3;
  #glance: [number, number] = [0, 0];
  #twitchAt = 4;
  #twitch = -1;
  #eased: [number, number] = [0, 0];
  #pet = 0;

  constructor(params: readonly ParamRange[], options: PerformerOptions = {}) {
    for (const p of params) this.#ranges.set(p.name, p);
    this.#random = options.random ?? Math.random;
  }

  get expression(): Expression { return this.#expression; }
  get motion(): Motion | undefined { return this.#clip?.motion; }

  /** Holds an expression for `seconds`, or until replaced; `Infinity` pins it. */
  express(expression: Expression, seconds = 4.5): void {
    this.#expression = expression;
    this.#expressionLeft = expression === "neutral" ? 0 : seconds;
  }

  /** Starts a motion clip, replacing the one in flight. */
  play(motion: Motion): void {
    const length = { wave: 2.4, nod: 1.3, shake: 1.5, tail: 2.2, fins: 1.4, talk: 2.6, sway: 3.2, cheer: 2.2 }[motion];
    this.#clip = { motion, t: 0, length };
  }

  /** A tap somewhere on the character. Returns the expression it settles into. */
  touch(touch: Touch): Expression {
    const length = { head: 1.6, face: 1.3, fin: 1.2, tail: 1.6, body: 1.2, hand: 0.4, ahoge: 1.2, skirt: 1.2 }[touch];
    this.#reaction = { touch, t: 0, length };
    const next: Expression = { head: "happy", face: "surprised", fin: "shy", tail: "angry", body: "shy", hand: "happy", ahoge: "surprised", skirt: "angry" }[touch] as Expression;
    this.express(next, touch === "hand" ? 2.6 : 2.4);
    if (touch === "hand") this.play("wave");
    if (touch === "tail") this.play("tail");
    if (touch === "fin") this.play("fins");
    return next;
  }

  /** Advances `dt` seconds and returns every param this model carries, by id. */
  step(dt: number): Map<string, number> {
    dt = clamp(dt, 0, 0.1);
    this.#time += dt;
    const t = this.#time;
    const target: Pose = {};
    const add = (name: string, value: number): void => { target[name] = (target[name] ?? 0) + value; };

    // -- idle: breath, weight shifting, a slow tail, the odd fin twitch
    if (this.idle) {
      add("breath", 0.5 - 0.5 * Math.cos(t * 2 * Math.PI / 3.8));
      add("body_move_x", 0.9 * Math.sin(t * 0.52) + 0.4 * Math.sin(t * 0.21 + 1));
      add("body_roll", 0.7 * Math.sin(t * 0.37 + 0.6));
      add("head_roll", 1.6 * Math.sin(t * 0.43 + 2));
      add("tail_wag", 0.38 * Math.sin(t * 1.25) + 0.12 * Math.sin(t * 2.9));
      if (t >= this.#twitchAt) { this.#twitch = 0; this.#twitchAt = t + 3.5 + this.#random() * 6; }
    }
    if (this.#twitch >= 0) {
      this.#twitch += dt / 0.42;
      if (this.#twitch >= 1) this.#twitch = -1;
      else add("fin_flap", Math.sin(this.#twitch * Math.PI * 2) * 0.8 * (1 - this.#twitch));
    }

    // -- where the eyes and head go
    let aim: [number, number];
    if (this.followPointer && this.look) aim = this.look;
    else {
      if (t >= this.#glanceAt) {
        const away = this.#random() < 0.6;
        this.#glance = away ? [(this.#random() * 2 - 1) * 0.7, (this.#random() * 2 - 1) * 0.35] : [0, 0];
        this.#glanceAt = t + 1.4 + this.#random() * 3.2;
      }
      aim = this.idle ? this.#glance : [0, 0];
    }
    const k = 1 - Math.exp(-dt / 0.16);
    this.#eased = [this.#eased[0] + (aim[0] - this.#eased[0]) * k, this.#eased[1] + (aim[1] - this.#eased[1]) * k];
    const [lx, ly] = this.#eased;
    add("head_yaw", 27 * lx); add("head_pitch", 22 * ly); add("head_roll", -5 * lx * (0.4 + Math.abs(ly)));
    add("body_yaw", 6.5 * lx); add("body_roll", 1.6 * lx); add("body_move_x", 2.2 * lx);
    // the eyes get there first, then settle as the head catches up
    add("gaze_x", clamp(aim[0] * 0.55 + (aim[0] - lx) * 1.6, -1, 1));
    add("gaze_y", clamp(aim[1] * 0.55 + (aim[1] - ly) * 1.6, -1, 1));

    // -- dragged about by the visitor
    if (this.drag) {
      add("body_move_x", 9 * this.drag[0]); add("body_roll", 8 * this.drag[0]);
      add("head_roll", -9 * this.drag[0]); add("head_pitch", 8 * this.drag[1]);
    }

    // -- expression
    if (this.#expressionLeft > 0) {
      this.#expressionLeft -= dt;
      if (this.#expressionLeft <= 0) this.#expression = "neutral";
    }
    const face: Pose = { ...EXPRESSIONS[this.#expression] };

    // -- being petted wins the face while it lasts
    this.#pet = clamp(this.#pet + (this.petting ? dt / 0.18 : -dt / 0.5), 0, 1);
    if (this.#pet > 0) {
      const p = smooth(this.#pet);
      face.eye_smile = Math.max(face.eye_smile ?? 0, p); face.eye_open = Math.min(face.eye_open ?? 1, 1 - p);
      face.face_blush = Math.max(face.face_blush ?? 0, 0.75 * p); face.mouth_expression = Math.max(face.mouth_expression ?? 0, p);
      face.mouth_open = Math.max(face.mouth_open ?? 0, 0.25 * p);
      add("head_pitch", -6 * p); add("head_roll", 5 * p * Math.sin(t * 5.5));
      add("fin_flap", 0.75 * p * Math.sin(t * 13)); add("tail_wag", 0.7 * p * Math.sin(t * 9));
    }

    // -- motion clip
    const clip = this.#clip;
    if (clip) {
      clip.t += dt;
      const u = clip.t / clip.length;
      const env = smooth(u / 0.18) * smooth((1 - u) / 0.22);
      switch (clip.motion) {
        case "wave":
          add("arm_left_raise", env * (0.84 + 0.16 * Math.sin(clip.t * 15)));
          add("head_roll", -7 * env); add("body_roll", 2.5 * env);
          face.mouth_expression = Math.max(face.mouth_expression ?? 0, env); face.mouth_open = Math.max(face.mouth_open ?? 0, 0.4 * env);
          break;
        case "cheer":
          add("arm_left_raise", env * (0.9 + 0.1 * Math.sin(clip.t * 12))); add("arm_right_raise", env * (0.9 + 0.1 * Math.sin(clip.t * 12 + 1.4)));
          add("head_pitch", 9 * env); add("tail_wag", env * Math.sin(clip.t * 16));
          face.mouth_open = Math.max(face.mouth_open ?? 0, 0.8 * env); face.mouth_expression = Math.max(face.mouth_expression ?? 0, env);
          face.eye_smile = Math.max(face.eye_smile ?? 0, env); face.eye_open = Math.min(face.eye_open ?? 1, 1 - env);
          break;
        case "nod": add("head_pitch", -17 * env * Math.max(0, Math.sin(clip.t * 10.5))); break;
        case "shake": add("head_yaw", 19 * env * Math.sin(clip.t * 11)); break;
        case "tail": add("tail_wag", env * Math.sin(clip.t * 17)); add("body_roll", 1.6 * env * Math.sin(clip.t * 8.5)); break;
        case "fins": add("fin_flap", env * Math.sin(clip.t * 21)); break;
        case "sway":
          add("body_roll", 7 * env * Math.sin(clip.t * 3.9)); add("body_move_x", 7 * env * Math.sin(clip.t * 3.9));
          add("head_roll", -9 * env * Math.sin(clip.t * 3.9 + 0.5)); add("tail_wag", env * Math.sin(clip.t * 3.9 + 1));
          face.mouth_expression = Math.max(face.mouth_expression ?? 0, env);
          break;
        case "talk": {
          const beat = Math.abs(Math.sin(clip.t * 11.5)) * (0.55 + 0.45 * Math.sin(clip.t * 3.7 + 1) ** 2);
          face.mouth_open = Math.max(face.mouth_open ?? 0, env * beat); add("head_pitch", 2.5 * env * Math.sin(clip.t * 5));
          break;
        }
      }
      if (u >= 1) this.#clip = undefined;
    }

    // -- a flinch where they poked
    const hit = this.#reaction;
    if (hit) {
      hit.t += dt;
      const u = hit.t / hit.length;
      const pop = Math.exp(-u * 5) * Math.sin(u * 22);
      switch (hit.touch) {
        case "face": add("head_roll", 9 * pop); add("head_pitch", 6 * pop); break;
        case "head": add("head_pitch", -8 * Math.exp(-u * 3)); break;
        case "ahoge": add("head_roll", 11 * pop); break;
        case "body": case "skirt": add("body_roll", 5 * pop); add("body_move_x", 5 * pop); break;
        case "tail": add("body_yaw", 7 * pop); break;
        case "fin": add("head_roll", -8 * pop); break;
        case "hand": break;
      }
      if (u >= 1) this.#reaction = undefined;
    }

    if (this.voice !== undefined) face.mouth_open = Math.max(face.mouth_open ?? 0, clamp(this.voice, 0, 1));

    // -- blink
    let lid = 1;
    if (this.autoBlink && this.#blink < 0 && t >= this.#blinkAt) { this.#blink = 0; this.#again = this.#random() < 0.18; }
    if (this.#blink >= 0) {
      this.#blink += dt;
      const b = this.#blink;
      lid = b < 0.07 ? 1 - b / 0.07 : b < 0.12 ? 0 : b < 0.26 ? (b - 0.12) / 0.14 : 1;
      if (b >= 0.26) {
        this.#blink = -1;
        this.#blinkAt = t + (this.#again ? 0.12 : (this.#expression === "sleepy" ? 1.2 : 1.9) + this.#random() * 4);
        this.#again = false;
      }
    }

    // -- fan the face out to the rig's left and right params
    add("head_pitch", face.pitch ?? 0); add("head_roll", face.roll ?? 0);
    add("gaze_x", face.gaze_x ?? 0); add("gaze_y", face.gaze_y ?? 0);
    const open = face.eye_open ?? 1;
    const out: Pose = { ...target };
    for (const side of ["left", "right"] as const) {
      const winking = side === "left" ? face.wink ?? 0 : 0;
      out[`eye_${side}_open`] = Math.min(open * (1 - winking), lid);
      out[`eye_${side}_smile`] = Math.max(face.eye_smile ?? 0, winking);
      out[`eye_${side}_gaze_x`] = clamp(target.gaze_x ?? 0, -1, 1);
      out[`eye_${side}_gaze_y`] = clamp(target.gaze_y ?? 0, -1, 1);
      out[`eyebrow_${side}_height`] = face.brow_height ?? 0;
      out[`eyebrow_${side}_inner_height`] = face.brow_inner ?? 0;
      out[`cheek_raise_${side}`] = face.eye_smile ?? 0;
    }
    delete out.gaze_x; delete out.gaze_y;
    out.mouth_open = face.mouth_open ?? 0;
    out.mouth_expression = face.mouth_expression ?? 0;
    out.face_blush = face.face_blush ?? 0;

    // -- ease, clamp, and hand back what this model actually has
    const result = new Map<string, number>();
    for (const [name, range] of this.#ranges) {
      // A param a layer drove once and has let go of eases home rather than
      // freezing where the clip left it; one never driven is not this file's.
      const wanted = out[name] ?? (this.#current.has(name) ? range.default : undefined);
      if (wanted === undefined) continue;
      const goal = clamp(name === "face_blush" && range.default > 0 ? Math.max(wanted, range.default) : wanted, range.min, range.max);
      const tau = easeOf(name);
      const from = this.#current.get(name) ?? range.default;
      const value = tau === 0 ? goal : from + (goal - from) * (1 - Math.exp(-dt / tau));
      this.#current.set(name, value);
      result.set(range.id, value);
    }
    return result;
  }
}

function easeOf(name: string): number {
  if (name.endsWith("_open") && name.startsWith("eye_")) return 0;   // a blink is not eased
  if (name.startsWith("head_")) return EASE.head;
  if (name.startsWith("body_") || name === "breath") return EASE.body;
  if (name.includes("_gaze_")) return EASE.gaze;
  if (name.startsWith("arm_") || name === "tail_wag" || name === "fin_flap") return EASE.limb;
  return EASE.face;
}
