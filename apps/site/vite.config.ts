import { stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { localExample } from "./local-example.js";

// `packages/wasm` sits outside this app, and the dev server serves only what it
// has been allowed.
const workspaceRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig(async ({ command, isPreview }) => {
  const path = command === "serve" && !isPreview
    ? process.env.CATCHLIGHT_LOCAL_MODEL
    : undefined;
  const local = path ? await localExample(path) : undefined;
  return {
    // GitHub Pages serves under /<repo>/; a dev server and `preview` serve under /.
    base: process.env.VITE_BASE ?? "/",
    define: {
      // Fetch exposes decoded bytes; HTTP Content-Length may describe compression.
      __CATCHLIGHT_MODEL_SIZES__: JSON.stringify({
        lan: (await stat(new URL("../../tests/models/lan.clm", import.meta.url))).size,
        "dafeiyu-q": (await stat(new URL("./models/dafeiyu-q.clm", import.meta.url))).size,
        "dafeiyu-adult": (await stat(new URL("./models/dafeiyu-adult.clm", import.meta.url))).size,
      }),
      __CATCHLIGHT_EXAMPLE__: JSON.stringify(local?.example ?? null),
    },
    plugins: [react(), ...(local ? [local.plugin] : [])],
    server: {
      ...(local ? { host: "127.0.0.1" } : {}),
      fs: { allow: [workspaceRoot] },
    },
    build: {
      // The wasm bundle is ~28 MiB in debug and megabytes in release; the warning
      // says nothing this build does not already know.
      chunkSizeWarningLimit: 8192,
    },
  };
});
