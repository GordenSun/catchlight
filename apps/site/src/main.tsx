/** The entry page stays lightweight; a demo loads only after selection. */
import { createRoot } from "react-dom/client";
import logo from "../../../assets/logo-transparent.svg";
import "./site.css";
import { showLoading } from "./loading.js";

const query = new URLSearchParams(location.search);
const named = query.get("demo");
// Existing connected and probe URLs remain direct editor entry points. With
// nothing named, the page is the interactive character: a visitor lands on
// something that moves, and the chooser is one link away at `?demo=menu`.
const mode = named === "face" || named === "live" ? named
  : named === "editor" || query.has("server") || query.has("probe") ? "editor"
  : named === "menu" ? undefined : "live";
if (mode) {
  void import("./demo.js").then(({ start }) => start(mode)).catch(cause => {
    showLoading("Could not open Catchlight", "Reload the page to try again.");
    document.getElementById("loading-progress")?.setAttribute("hidden", "");
    const failure = document.getElementById("failure")!;
    failure.hidden = false;
    failure.textContent = `Could not load the demo. Reload to try again. ${String(cause)}`;
  });
} else {
  createRoot(document.getElementById("editor")!).render(
    <main className="chooser">
      <div className="demo-brand"><img src={logo} alt="" width="40" height="40" />catchlight</div>
      <p className="eyebrow">CHARACTERS IN MOTION</p>
      <h1>Bring a character to life.</h1>
      <p className="intro">Three ways to explore Catchlight’s 2.5D animation. Pick a demo to begin.</p>
      <div className="demo-cards">
        <a className="demo-card demo-card-wide" href="?demo=live">
          <span className="card-number">00 / PLAY</span><h2>大肥鱼 · Interactive character</h2>
          <p>Pat her head, poke a cheek, tug the tail. Two rigs of one character, a chibi and a grown-up, switch on one page — no camera needed.</p>
          <strong>Meet 大肥鱼 <span aria-hidden="true">↗</span></strong>
        </a>
        <a className="demo-card" href="?demo=editor">
          <span className="card-number">01 / CREATE</span><h2>Editor demo</h2>
          <p>Explore Lan’s rig, pose the character, and shape your own animation in the studio.</p>
          <strong>Open the editor <span aria-hidden="true">↗</span></strong>
        </a>
        <a className="demo-card" href="?demo=face">
          <span className="card-number">02 / PERFORM</span><h2>Face tracking</h2>
          <p>Turn your head, blink, and speak. Watch Lan follow your expressions through your camera.</p>
          <strong>Try face tracking <span aria-hidden="true">↗</span></strong>
        </a>
      </div>
      <p className="chooser-note">Runs in your browser. Face tracking needs a camera; the others do not. Add <code>&amp;model=dafeiyu-q</code> or <code>dafeiyu-adult</code> to open 大肥鱼 in the editor or under face tracking.</p>
    </main>,
  );
}
