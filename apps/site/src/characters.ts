/** The characters this site ships, and how each one is framed.
 *
 * A model file says nothing about cameras, so the numbers a page needs to put
 * a face in the middle of a canvas live here, beside the files they describe.
 * World units, Y up; `head` is the centre of the head and its radius. */
import type { Camera } from "@catchlight/core";

import lanUrl from "../../../tests/models/lan.clm?url";
import adultUrl from "../models/dafeiyu-adult.clm?url";
import chibiUrl from "../models/dafeiyu-q.clm?url";

export type CharacterId = "lan" | "dafeiyu-q" | "dafeiyu-adult";

export interface Character {
  id: CharacterId;
  /** The session's file name, which is also what the editor titles it: ASCII, because a store key is. */
  file: string;
  url: string;
  /** Decoded bytes, for a progress bar that survives HTTP compression. */
  size: number;
  /** `[min_x, min_y, max_x, max_y]` of the art at rest. */
  bounds: [number, number, number, number];
  head: { x: number; y: number; radius: number };
  /** Head and shoulders, where face tracking and close-ups look. */
  face: Camera;
  /** Waist up. */
  bust: Camera;
}

/** Sizes are measured at build time; see `vite.config.ts`. */
declare const __CATCHLIGHT_MODEL_SIZES__: Record<CharacterId, number>;

export const CHARACTERS: Record<CharacterId, Character> = {
  lan: {
    id: "lan", file: "Lan.clm", url: lanUrl, size: __CATCHLIGHT_MODEL_SIZES__.lan,
    bounds: [-190, 0, 190, 905], head: { x: -3, y: 790, radius: 62 },
    face: { center: [-3, 770], height: 520 }, bust: { center: [-3, 700], height: 640 },
  },
  "dafeiyu-q": {
    id: "dafeiyu-q", file: "dafeiyu-q.clm", url: chibiUrl, size: __CATCHLIGHT_MODEL_SIZES__["dafeiyu-q"],
    bounds: [-410, 0, 485, 1102], head: { x: 0, y: 684, radius: 121 },
    face: { center: [0, 700], height: 470 }, bust: { center: [10, 640], height: 800 },
  },
  "dafeiyu-adult": {
    id: "dafeiyu-adult", file: "dafeiyu-adult.clm", url: adultUrl, size: __CATCHLIGHT_MODEL_SIZES__["dafeiyu-adult"],
    bounds: [-325, 0, 415, 1409], head: { x: 0, y: 1229, radius: 48 },
    face: { center: [0, 1232], height: 300 }, bust: { center: [0, 1080], height: 720 },
  },
};

/** The character a URL names with `?model=`, or `fallback`. */
export function characterFromQuery(query: URLSearchParams, fallback: CharacterId): Character {
  const named = query.get("model");
  return CHARACTERS[named !== null && named in CHARACTERS ? named as CharacterId : fallback];
}

/** The whole figure with a little air, on a canvas of this shape. */
export function fullCamera(character: Character, aspect: number): Camera {
  const [x0, y0, x1, y1] = character.bounds;
  const height = Math.max((y1 - y0) * 1.1, (x1 - x0) * 1.08 / Math.max(aspect, 0.2));
  return { center: [(x0 + x1) / 2, (y0 + y1) / 2], height };
}
