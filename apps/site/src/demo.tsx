/** Boots a selected demo. Face tracking uses an ephemeral in-tab model. */

import {
  ConnectedBackend,
  Editor,
  InTabBackend,
  MemoryStorage,
  OpfsStorage,
} from "@catchlight/core";
import type { Backend, Session, Storage } from "@catchlight/core";
import { CatchlightEditor } from "@catchlight/editor-ui";
// Only the probe door below uses this, and only when the URL asks for it.
import { EditorProvider, fitCamera } from "@catchlight/react";
import "@catchlight/editor-ui/theme.css";
// The generated module initializes itself on import: `cargo xtask build-wasm` emits
// wasm-bindgen's bundler target, whose `.wasm` is an ESM import rather than a
// fetch this page would have to sequence.
import * as catchlight from "@catchlight/wasm";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { CHARACTERS, characterFromQuery } from "./characters.js";
import type { Character } from "./characters.js";
import { downloadModel, showLoading, showProcessing } from "./loading.js";
import { FaceDemo } from "./face-demo.js";
import { LiveDemo } from "./live-demo.js";

/** A bundled asset URL or development route, never a filesystem path. */
type Example = { file: string; name: string; size?: number };
declare const __CATCHLIGHT_EXAMPLE__: Example | null;

export type DemoMode = "editor" | "face" | "live";

/** The character a mode opens when the URL does not name one with `?model=`. */
const DEFAULT_CHARACTER = { editor: "lan", face: "lan", live: "dafeiyu-q" } as const;

export async function start(mode: DemoMode): Promise<void> {
  try {
    const query = new URLSearchParams(globalThis.location.search);
    const server = mode === "editor" ? query.get("server") : null;
    const character = characterFromQuery(query, DEFAULT_CHARACTER[mode]);
    // The development override replaces Lan in the editor and nothing else.
    const example: Example = mode === "editor" && character.id === "lan" && __CATCHLIGHT_EXAMPLE__
      ? __CATCHLIGHT_EXAMPLE__ : starterOf(character);
    if (mode === "live") showLoading("正在准备舞台…", "加载渲染引擎");
    else showLoading("Starting editor…", "Preparing your workspace");
    // Posing demos keep nothing: an in-tab editor over memory, gone with the tab.
    const editor = await Editor.create(catchlight, mode === "editor" ? await pick(server) : new InTabBackend(new catchlight.CatchlightEditor(), new MemoryStorage()));

    // A configured starter is opened before the panel chooses its current session.
    const session = server === null
      ? await openStarter(editor, example, mode === "live").catch(report)
      : undefined;

    mount(editor, mode, session, character, example);
    probe(editor);
  } catch (cause) {
    showLoading("Could not open Catchlight", "Reload the page to try again.");
    document.getElementById("loading-progress")?.setAttribute("hidden", "");
    report(cause);
  }
}

function starterOf(character: Character): Example {
  return { file: character.url, name: character.file, size: character.size };
}

/** `BASE_URL` ends in a slash and is `/<repo>/` on Pages. */
function assetUrl(file: string): string {
  return new URL(file, new URL(import.meta.env.BASE_URL, location.href)).href;
}

/** A local editor process when the URL names one, this tab otherwise. */
async function pick(server: string | null): Promise<Backend> {
  if (server !== null) return ConnectedBackend.connect(server);
  return new InTabBackend(new catchlight.CatchlightEditor(), await store());
}

/**
 * Where an in-tab editor's bytes live.
 *
 * OPFS survives a reload and a private window does not have it, so the
 * fallback keeps the editor usable rather than refusing to start.
 */
async function store(): Promise<Storage> {
  if (!OpfsStorage.available()) return new MemoryStorage();
  try {
    return await OpfsStorage.open();
  } catch (cause) {
    report(cause);
    return new MemoryStorage();
  }
}

/**
 * Opens the starter character, or the configured development override.
 *
 * Reported rather than thrown: a starter that will not load is worth saying out
 * loud, and is no reason to withhold an editor that works.
 */
async function openStarter(editor: Editor, example: Example, chinese = false): Promise<Session> {
  const bytes = await downloadModel(assetUrl(example.file), example.name, example.size, chinese ? "正在请大肥鱼出场…" : undefined);
  await showProcessing(example.name, chinese ? "正在整理模型和贴图…" : undefined);
  return editor.openFile(bytes, example.name);
}

/** Fetches one more character into an editor that is already on screen. */
async function openCharacter(editor: Editor, character: Character, progress: (fraction: number) => void): Promise<Session> {
  const response = await fetch(assetUrl(character.url));
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const bytes = new Uint8Array(character.size);
  let received = 0;
  const reader = response.body?.getReader();
  if (!reader) return editor.openFile(new Uint8Array(await response.arrayBuffer()), character.file);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > bytes.length) throw new Error(`${character.file} is larger than the build measured`);
    bytes.set(value, received);
    received += value.length;
    progress(received / character.size);
  }
  return editor.openFile(received === bytes.length ? bytes : bytes.subarray(0, received), character.file);
}

function mount(editor: Editor, mode: DemoMode, session: Session | void, character: Character, example: Example): void {
  const host = document.getElementById("editor");
  if (!host) throw new Error("index.html has no #editor to mount into");
  const menu = `${import.meta.env.BASE_URL}?demo=menu`;
  if (mode === "live") {
    document.documentElement.lang = "zh-CN";
    document.title = "大肥鱼 Live2D — 摸摸头、戳戳脸，Q版 / 成年版随时切换";
    createRoot(host).render(
      <StrictMode>
        {session ? <LiveDemo editor={editor} initial={character} session={session} load={(next, progress) => openCharacter(editor, next, progress)} /> : null}
      </StrictMode>,
    );
    return;
  }
  createRoot(host).render(
    <StrictMode>
      <div className="demo-shell">
        <nav className="demo-nav"><a href={menu}>← All demos</a><span>{mode === "face" ? "Face tracking" : "Editor demo"}</span>
          {character.id !== "lan" && <a href={`${import.meta.env.BASE_URL}?model=${character.id}`}>大肥鱼互动页</a>}</nav>
        {mode === "face" ? (session ? <EditorProvider editor={editor}><FaceDemo session={session} camera={character.face} name={character.id === "lan" ? "Lan" : "大肥鱼"} /></EditorProvider> : null) :
          <CatchlightEditor editor={editor} onOpenStarter={() => openStarter(editor, example)} />}
      </div>
    </StrictMode>,
  );
}

/**
 * A door for the browser smoke test, shut unless the URL asks for it.
 *
 * `?probe=1` puts the editor on `globalThis` so a driver in the page can
 * attach a second viewport. That is the one thing about the WebGL2 tier no
 * unit test can show — an extra canvas borrows the main one's surface, and
 * whether the main canvas survives that is a question only a real browser
 * answers. Nothing reaches this on an ordinary load, and nothing in the editor
 * reads it.
 *
 * `fitCamera` rides along because the test's second viewport has to be framed
 * the way the editor frames the first. It is the same function the React
 * viewport calls when it opens a model, so a driver that calls it with the
 * same bounds and the same size gets the same camera, and the two canvases can
 * then be compared pixel for pixel.
 */
function probe(editor: Editor): void {
  if (!new URLSearchParams(globalThis.location.search).has("probe")) return;
  (globalThis as unknown as Record<string, unknown>).__catchlightProbe = {
    editor,
    fitCamera,
  };
}

/** One `<pre>`, appended to. Several things can go wrong on one load. */
function report(cause: unknown): void {
  const pre = document.getElementById("failure");
  console.error(cause);
  if (!pre) return;
  pre.hidden = false;
  pre.textContent = `${pre.textContent ?? ""}${describe(cause)}\n`;
}

function describe(cause: unknown): string {
  if (cause instanceof Error) {
    return cause.cause === undefined ? cause.message : `${cause.message}: ${describe(cause.cause)}`;
  }
  return String(cause);
}
