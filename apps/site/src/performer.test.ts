import { describe, expect, test } from "bun:test";

import { Performer } from "./performer.js";
import type { ParamRange } from "./performer.js";

const range = (name: string, min: number, max: number, def = 0): ParamRange => ({ id: `p:${name}`, name, min, max, default: def });
const RIG = [
  range("head_yaw", -30, 30), range("head_pitch", -30, 30), range("head_roll", -30, 30), range("body_roll", -10, 10),
  range("eye_left_open", 0, 1, 1), range("eye_right_open", 0, 1, 1), range("eye_left_smile", 0, 1), range("eye_left_gaze_x", -1, 1),
  range("mouth_open", 0, 1), range("mouth_expression", -1, 1), range("face_blush", 0, 1), range("arm_left_raise", 0, 1),
  range("tail_wag", -1, 1), range("breath", 0, 1),
];
/** A fixed sequence, so blinks and glances land on the same frames every run. */
const steady = () => { let i = 0; return () => [0.31, 0.72, 0.05, 0.93, 0.48][i++ % 5]!; };
const run = (p: Performer, seconds: number, each?: (pose: Map<string, number>) => void) => {
  let last = new Map<string, number>();
  for (let i = 0; i < seconds * 60; i++) { last = p.step(1 / 60); each?.(last); }
  return last;
};

describe("Performer", () => {
  test("returns only params the rig carries, inside their ranges", () => {
    const p = new Performer(RIG, { random: steady() });
    p.look = [3, -3];
    p.play("wave"); p.express("happy");
    run(p, 6, pose => {
      expect([...pose.keys()].every(id => RIG.some(r => r.id === id))).toBe(true);
      for (const r of RIG) {
        const v = pose.get(r.id);
        if (v !== undefined) { expect(v).toBeGreaterThanOrEqual(r.min - 1e-9); expect(v).toBeLessThanOrEqual(r.max + 1e-9); }
      }
    });
  });

  test("the head turns toward the pointer and comes back when it leaves", () => {
    const p = new Performer(RIG, { random: steady() });
    p.idle = false; p.autoBlink = false;
    p.look = [1, 0.5];
    const toward = run(p, 1.5);
    expect(toward.get("p:head_yaw")!).toBeGreaterThan(20);
    expect(toward.get("p:head_pitch")!).toBeGreaterThan(8);
    p.look = undefined;
    expect(Math.abs(run(p, 2).get("p:head_yaw")!)).toBeLessThan(0.5);
  });

  test("the eyes lead the head", () => {
    const p = new Performer(RIG, { random: steady() });
    p.idle = false; p.autoBlink = false;
    p.look = [1, 0];
    const early = run(p, 0.1);
    expect(early.get("p:eye_left_gaze_x")!).toBeGreaterThan(0.6);
    expect(early.get("p:head_yaw")!).toBeLessThan(15);
  });

  test("blinks close both eyes fully and reopen them", () => {
    const p = new Performer(RIG, { random: steady() });
    let lowest = 1;
    const end = run(p, 8, pose => { lowest = Math.min(lowest, pose.get("p:eye_left_open")!); });
    expect(lowest).toBe(0);
    expect(end.get("p:eye_right_open")).toBe(end.get("p:eye_left_open"));
    p.autoBlink = false;
    run(p, 1);
    let shut = false;
    run(p, 8, pose => { shut ||= pose.get("p:eye_left_open")! < 1; });
    expect(shut).toBe(false);
  });

  test("an expression holds, then relaxes to neutral", () => {
    const p = new Performer(RIG, { random: steady() });
    p.autoBlink = false; p.idle = false;
    p.express("shy", 1);
    expect(run(p, 0.8).get("p:face_blush")!).toBeGreaterThan(0.9);
    expect(run(p, 1.5).get("p:face_blush")!).toBeLessThan(0.05);
    expect(p.expression).toBe("neutral");
  });

  test("a wink closes one eye only", () => {
    const p = new Performer(RIG, { random: steady() });
    p.autoBlink = false; p.idle = false;
    p.express("wink", Infinity);
    const pose = run(p, 1);
    expect(pose.get("p:eye_left_open")).toBe(0);
    expect(pose.get("p:eye_right_open")).toBe(1);
  });

  test("touching a hand waves and the clip ends by itself", () => {
    const p = new Performer(RIG, { random: steady() });
    p.idle = false;
    expect(p.touch("hand")).toBe("happy");
    expect(p.motion).toBe("wave");
    let peak = 0;
    const end = run(p, 3.2, pose => { peak = Math.max(peak, pose.get("p:arm_left_raise")!); });
    expect(peak).toBeGreaterThan(0.8);
    expect(p.motion).toBeUndefined();
    expect(end.get("p:arm_left_raise")!).toBeLessThan(0.02);
  });

  test("a voice level opens the mouth", () => {
    const p = new Performer(RIG, { random: steady() });
    p.idle = false; p.voice = 0.8;
    expect(run(p, 0.5).get("p:mouth_open")!).toBeGreaterThan(0.7);
    p.voice = undefined;
    expect(run(p, 0.6).get("p:mouth_open")!).toBeLessThan(0.02);
  });
});
