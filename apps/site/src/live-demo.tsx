/** 大肥鱼, alive in the tab: one page, two rigs, no camera needed.
 *
 * The page poses an ephemeral in-tab model and never edits it. A `Performer`
 * turns the pointer, a clock and the buttons into param values each frame; the
 * model's own motion responses add the hair, fins, tail and skirt lag on top.
 *
 * **Both characters stay open once loaded.** Switching swaps the session on
 * one canvas, so the second switch is instant and the camera never rebuilds.
 * The other character is fetched in the background after the first is up.
 *
 * **A touch is classified in world space, not by the part under it.** Parts
 * are meshed as grids over their art, so the front hair "covers" the whole
 * face as far as picking goes. The head is a circle the page already knows;
 * only what falls outside it is asked of `pickNode`.
 *
 * **The microphone is opt-in and only ever a level.** One analyser, one
 * number per frame, nothing recorded or sent; turning it off or leaving the
 * tab stops the track.
 */
import type { Camera, Editor, Session } from "@catchlight/core";
import { EditorProvider, screenAt, Viewport } from "@catchlight/react";
import type { Size, ViewportPointerEvent } from "@catchlight/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CHARACTERS, fullCamera } from "./characters.js";
import type { Character, CharacterId } from "./characters.js";
import { TierBadge, useGpuTier } from "./face-stats.js";
import { Performer } from "./performer.js";
import type { Expression, Motion, Touch } from "./performer.js";

type Shot = "full" | "bust" | "face";
const CAST: { id: CharacterId; label: string; blurb: string }[] = [
  { id: "dafeiyu-q", label: "Q版", blurb: "二头身，软乎乎" },
  { id: "dafeiyu-adult", label: "成年版", blurb: "长裙女仆，温柔款" },
];
const EXPRESSIONS: [Expression, string][] = [
  ["happy", "开心"], ["shy", "害羞"], ["angry", "生气"], ["surprised", "惊讶"], ["sleepy", "犯困"], ["wink", "眨眼"], ["smug", "得意"],
];
const MOTIONS: [Motion, string][] = [
  ["wave", "挥手"], ["cheer", "欢呼"], ["nod", "点头"], ["shake", "摇头"], ["tail", "摇尾巴"], ["fins", "抖鱼鳍"], ["sway", "晃一晃"], ["talk", "说话"],
];
const SHOTS: [Shot, string][] = [["full", "全身"], ["bust", "半身"], ["face", "特写"]];
/** Sky blue behind the character, the same colour the page paints around the canvas. */
const BACKDROP: [number, number, number] = [0.878, 0.949, 1];

const TOUCH_LINES: Record<Touch, string[]> = {
  head: ["嘿嘿，再摸摸~", "好舒服呀~", "本鱼最喜欢被摸头了！"],
  face: ["呜哇！不要戳脸啦！", "脸要被戳扁了…", "欸？！"],
  fin: ["鱼鳍很敏感的！", "痒痒痒~", "不、不可以捏鱼鳍…"],
  tail: ["尾巴不能乱拉！", "再拉本鱼要生气了哦！", "哼！"],
  body: ["你、你在摸哪里呀…", "好害羞…", "女仆装好看吗？"],
  skirt: ["不许掀裙子！", "大笨蛋！"],
  hand: ["你好呀~", "嗨！本鱼在这里！", "击掌！"],
  ahoge: ["呆毛要被拔掉啦！", "那是本鱼的天线！"],
};
const MOOD_LINES: Record<Expression, string[]> = {
  neutral: [], happy: ["今天也超开心！", "欸嘿嘿~"], shy: ["别、别一直盯着看…"], angry: ["本鱼生气了！哼！"],
  surprised: ["真的假的？！"], sleepy: ["呼啊…好困…"], wink: ["给你一个 wink~"], smug: ["哼哼，不愧是本鱼。"],
};
const MOTION_LINES: Record<Motion, string[]> = {
  wave: ["你好呀~"], cheer: ["好耶！！"], nod: ["嗯嗯！"], shake: ["不要不要~"], tail: ["看本鱼的尾巴！"], fins: ["扑棱扑棱~"],
  sway: ["啦啦啦~"], talk: ["我是大肥鱼，请多关照！", "今天也要元气满满哦！"],
};

export interface LiveDemoProps {
  /** The character to show first, already opened. */
  initial: Character;
  session: Session;
  /** Fetches and opens another character; the page reports progress itself. */
  load: (character: Character, progress: (fraction: number) => void) => Promise<Session>;
  editor: Editor;
}

export function LiveDemo(props: LiveDemoProps) {
  return <EditorProvider editor={props.editor}><Stage {...props} /></EditorProvider>;
}

function Stage({ initial, session: first, load, editor }: LiveDemoProps) {
  const tier = useGpuTier(editor);
  const [who, setWho] = useState<CharacterId>(initial.id);
  const [sessions, setSessions] = useState<Partial<Record<CharacterId, Session>>>({ [initial.id]: first });
  const [loading, setLoading] = useState<{ id: CharacterId; fraction: number } | undefined>();
  const [error, setError] = useState("");
  const [shot, setShot] = useState<Shot>("full");
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  const [camera, setCamera] = useState<Camera | undefined>();
  const [bubble, setBubble] = useState<{ text: string; key: number } | undefined>();
  const [mood, setMood] = useState<Expression>("neutral");
  const [follow, setFollow] = useState(true);
  const [blink, setBlink] = useState(true);
  const [idle, setIdle] = useState(true);
  const [mic, setMic] = useState(false);
  // On a phone the panel would take half the screen from her; it starts shut there.
  const [open, setOpen] = useState(() => globalThis.matchMedia?.("(min-width: 761px)").matches ?? true);

  const character = CHARACTERS[who];
  const session = sessions[who];
  const performer = useMemo(() => session ? new Performer(session.params()) : undefined, [session]);
  const pending = useRef(new Map<CharacterId, Promise<Session>>());

  /** Opens a character once, however many callers ask while it is on its way. */
  const fetchCharacter = useCallback((id: CharacterId, visible: boolean): Promise<Session> => {
    let job = pending.current.get(id);
    if (!job) {
      job = load(CHARACTERS[id], fraction => setLoading(now => now?.id === id ? { id, fraction } : now));
      pending.current.set(id, job);
      void job.then(opened => setSessions(all => ({ ...all, [id]: opened })), () => pending.current.delete(id));
    }
    if (visible) setLoading({ id, fraction: 0 });
    return job;
  }, [load]);

  const choose = (id: CharacterId): void => {
    setError("");
    setWho(id);
    if (sessions[id]) return;
    fetchCharacter(id, true)
      .catch((cause: unknown) => setError(`加载失败：${cause instanceof Error ? cause.message : String(cause)}`))
      .finally(() => setLoading(now => now?.id === id ? undefined : now));
  };

  // The other character, fetched quietly once this one is on screen.
  useEffect(() => {
    const other = CAST.find(c => c.id !== initial.id)?.id;
    if (!other) return;
    const timer = setTimeout(() => { void fetchCharacter(other, false).catch(() => undefined); }, 2500);
    return () => clearTimeout(timer);
  }, [initial.id, fetchCharacter]);

  // -- camera: eased toward the chosen shot; a wheel zoom moves it directly
  const aspect = size.height > 0 ? size.width / size.height : 1;
  const wanted = useMemo<Camera>(() => {
    if (shot === "full") return fullCamera(character, aspect);
    const base = shot === "bust" ? character.bust : character.face;
    // a narrow canvas needs more height to keep the same width in view
    return { center: base.center, height: base.height * Math.max(1, 0.78 / Math.max(aspect, 0.2)) };
  }, [shot, character, aspect]);
  const goal = useRef(wanted);
  goal.current = wanted;
  const live = useRef<Camera | undefined>(undefined);
  useEffect(() => {
    let frame = 0;
    const tick = (): void => {
      const to = goal.current, from = live.current;
      if (!from) { live.current = to; setCamera(to); }
      else {
        const dx = to.center[0] - from.center[0], dy = to.center[1] - from.center[1], dh = to.height - from.height;
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dh) > 0.05) {
          const next: Camera = { center: [from.center[0] + dx * 0.16, from.center[1] + dy * 0.16], height: from.height + dh * 0.16 };
          live.current = next; setCamera(next);
        }
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  // -- the performance: one step a frame, written straight to the session
  const options = useRef({ follow, blink, idle });
  options.current = { follow, blink, idle };
  useEffect(() => {
    if (!session || !performer) return;
    let frame = 0, last = performance.now();
    const tick = (now: number): void => {
      performer.followPointer = options.current.follow;
      performer.autoBlink = options.current.blink;
      performer.idle = options.current.idle;
      for (const [id, value] of performer.step((now - last) / 1000)) session.setParam(id, value);
      last = now;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [session, performer]);
  useEffect(() => {
    if (!performer) return;
    const timer = setInterval(() => setMood(performer.expression), 200);
    return () => clearInterval(timer);
  }, [performer]);

  const say = useCallback((lines: readonly string[]): void => {
    if (lines.length) setBubble({ text: lines[Math.floor(Math.random() * lines.length)]!, key: Date.now() });
  }, []);
  useEffect(() => {
    if (!bubble) return;
    const timer = setTimeout(() => setBubble(undefined), 2600);
    return () => clearTimeout(timer);
  }, [bubble]);

  // -- pointer: where to look, what was touched, and being dragged about
  const stage = useRef<HTMLDivElement>(null);
  const press = useRef<{ touch: Touch | undefined; screen: [number, number]; moved: boolean; pointer: string } | undefined>(undefined);
  const lookAway = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const headScreen = (): [number, number] | undefined => {
    const cam = live.current;
    return cam && size.height ? screenAt(cam, size, [character.head.x, character.head.y]) : undefined;
  };
  const lookAt = (screen: [number, number]): void => {
    const at = headScreen();
    if (!performer || !at) return;
    const reach = Math.max(120, Math.min(size.width, size.height) * 0.45);
    performer.look = [clamp((screen[0] - at[0]) / reach), clamp((at[1] - screen[1]) / reach)];
  };
  const classify = (world: [number, number]): Touch | undefined => {
    const { x, y, radius } = character.head;
    const dx = world[0] - x, dy = world[1] - y;
    if (Math.hypot(dx, dy) < radius * 1.02) return dy > radius * 0.4 ? "head" : "face";
    const node = session?.pickNode(world[0], world[1]);
    if (!node) return undefined;
    if (node === "ahoge") return "ahoge";
    if (node.startsWith("fin-") || node.startsWith("bow-")) return "fin";
    if (node === "tail") return "tail";
    if (node.startsWith("arm-")) return "hand";
    if (node === "body") return world[1] < character.head.y - radius * 2.6 ? "skirt" : "body";
    if (node === "fronthair" && dy > 0) return "head";
    return undefined;
  };
  const onDown = ({ world, screen, event }: ViewportPointerEvent): void => {
    clearTimeout(lookAway.current);
    press.current = { touch: classify(world), screen, moved: false, pointer: event.pointerType };
    lookAt(screen);
  };
  const onMove = ({ screen }: ViewportPointerEvent): void => {
    lookAt(screen);
    const held = press.current;
    if (!held || !performer) return;
    const dx = screen[0] - held.screen[0], dy = screen[1] - held.screen[1];
    if (Math.hypot(dx, dy) > 10) held.moved = true;
    if (!held.moved) return;
    if (held.touch === "head") {
      if (!performer.petting) say(TOUCH_LINES.head);
      performer.petting = true;
    } else if (held.touch) performer.drag = [clamp(dx / 220), clamp(-dy / 220)];
  };
  const onUp = ({ event }: ViewportPointerEvent): void => {
    const held = press.current;
    press.current = undefined;
    if (!performer) return;
    performer.drag = undefined;
    const stroked = performer.petting;
    performer.petting = false;
    if (held?.touch && !held.moved) { performer.touch(held.touch); say(TOUCH_LINES[held.touch]); }
    else if (stroked) performer.express("happy", 2);
    // a finger lifted is not a pointer parked: let the gaze drift home
    if (event.pointerType !== "mouse") lookAway.current = setTimeout(() => { performer.look = undefined; }, 1500);
  };
  const onLeave = (): void => { if (performer && !press.current) performer.look = undefined; };

  // -- microphone lip-sync
  useEffect(() => {
    if (!mic || !performer) return;
    let stopped = false, frame = 0;
    let stream: MediaStream | undefined, audio: AudioContext | undefined;
    const stop = (): void => {
      stopped = true;
      cancelAnimationFrame(frame);
      stream?.getTracks().forEach(track => track.stop());
      void audio?.close();
      performer.voice = undefined;
    };
    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("这个浏览器不支持麦克风，或页面不是 HTTPS。");
      const got = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      if (stopped) { got.getTracks().forEach(track => track.stop()); return; }
      stream = got;
      audio = new AudioContext();
      const analyser = audio.createAnalyser();
      analyser.fftSize = 1024;
      audio.createMediaStreamSource(got).connect(analyser);
      const samples = new Float32Array(analyser.fftSize);
      let level = 0;
      const tick = (): void => {
        analyser.getFloatTimeDomainData(samples);
        let sum = 0;
        for (const s of samples) sum += s * s;
        const loud = clamp((Math.sqrt(sum / samples.length) - 0.012) * 9, 0, 1);
        level += (loud - level) * (loud > level ? 0.55 : 0.25);   // open fast, close softly
        performer.voice = level;
        frame = requestAnimationFrame(tick);
      };
      tick();
    })().catch((cause: unknown) => {
      setMic(false);
      setError(`麦克风没打开：${cause instanceof Error ? cause.message : String(cause)}`);
    });
    const hidden = (): void => { if (document.hidden) setMic(false); };
    document.addEventListener("visibilitychange", hidden);
    return () => { document.removeEventListener("visibilitychange", hidden); stop(); };
  }, [mic, performer]);

  const top = camera && size.height ? screenAt(camera, size, [character.head.x, character.head.y + character.head.radius * 1.9]) : undefined;
  const base = import.meta.env.BASE_URL;

  return <main className="live" data-open={open || undefined}>
    <header className="live-bar">
      <a className="live-brand" href={base}><b>大肥鱼</b><span>Live2D · Catchlight</span></a>
      <div className="live-cast" role="radiogroup" aria-label="选择角色">
        {CAST.map(c => <button key={c.id} type="button" role="radio" aria-checked={who === c.id} title={c.blurb} onClick={() => choose(c.id)}>{c.label}</button>)}
      </div>
      <nav className="live-links">
        <a href={`${base}?demo=face&model=${who}`}>摄像头面捕</a>
        <a href={`${base}?demo=editor&model=${who}`}>在编辑器里打开</a>
        <a href={`${base}?demo=menu`}>全部演示</a>
      </nav>
    </header>

    <div className="live-stage" ref={stage} onPointerLeave={onLeave}>
      <Viewport.Root session={session} {...(camera ? { camera } : {})} background={BACKDROP} wheelZoom={false}
        aria-label={`${CAST.find(c => c.id === who)?.label ?? ""}大肥鱼，可以摸头、戳脸、拉尾巴`}
        onResize={setSize} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        onError={cause => setError(String(cause))} />
      <div className="live-sea" aria-hidden="true">{Array.from({ length: 9 }, (_, i) => <i key={i} />)}</div>
      {bubble && top && <p className="live-bubble" key={bubble.key} role="status" style={{ left: top[0], top: Math.max(top[1], 12) }}>{bubble.text}</p>}
      {loading && loading.id === who && !session && <div className="live-loading" role="status">
        <span>正在请{CAST.find(c => c.id === who)?.label}大肥鱼出场… {Math.round(loading.fraction * 100)}%</span>
        <progress max={1} value={loading.fraction} />
      </div>}
      {error && <p className="live-error" role="alert">{error}</p>}
      <TierBadge tier={tier} />
      <p className="live-hint">摸摸头 · 戳戳脸 · 捏鱼鳍 · 拉尾巴 · 握握手 · 按住身体拖一拖</p>
      <button type="button" className="live-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "收起面板" : "表情 / 动作"}</button>
    </div>

    <aside className="live-panel" hidden={!open}>
      <section>
        <h2>表情</h2>
        <div className="live-grid">
          {EXPRESSIONS.map(([id, label]) => <button key={id} type="button" aria-pressed={mood === id}
            onClick={() => { const next = mood === id ? "neutral" : id; performer?.express(next, 6); setMood(next); say(MOOD_LINES[next]); }}>{label}</button>)}
        </div>
      </section>
      <section>
        <h2>动作</h2>
        <div className="live-grid">
          {MOTIONS.map(([id, label]) => <button key={id} type="button" onClick={() => { performer?.play(id); say(MOTION_LINES[id]); }}>{label}</button>)}
        </div>
      </section>
      <section>
        <h2>镜头</h2>
        <div className="live-grid live-shots" role="radiogroup" aria-label="镜头">
          {SHOTS.map(([id, label]) => <button key={id} type="button" role="radio" aria-checked={shot === id} onClick={() => setShot(id)}>{label}</button>)}
        </div>
      </section>
      <section className="live-switches">
        <label><input type="checkbox" role="switch" checked={follow} onChange={e => setFollow(e.target.checked)} />视线跟随指针</label>
        <label><input type="checkbox" role="switch" checked={blink} onChange={e => setBlink(e.target.checked)} />自动眨眼</label>
        <label><input type="checkbox" role="switch" checked={idle} onChange={e => setIdle(e.target.checked)} />待机小动作</label>
        <label><input type="checkbox" role="switch" checked={mic} onChange={e => { setError(""); setMic(e.target.checked); }} />麦克风对口型</label>
      </section>
      <p className="live-note">头发、鱼鳍、尾巴和裙摆的晃动是模型自带的弹簧物理。麦克风只在本机读取音量，不录音、不上传。</p>
    </aside>
  </main>;
}

function clamp(value: number, lo = -1, hi = 1): number {
  return Math.max(lo, Math.min(hi, value));
}
