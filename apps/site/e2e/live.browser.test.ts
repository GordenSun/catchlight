/** The interactive character page: it loads, follows the pointer, takes a
 * button and a touch, and switches rigs on one canvas. What the frame looks
 * like is the E2E driver's business; this reads the pose the page wrote. */
import { expect, test } from "bun:test";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";
import siteConfig from "../vite.config.js";

const executablePath = process.env.CHROMIUM ?? Bun.which("chromium");

test.skipIf(!executablePath)("the live page poses 大肥鱼 from the pointer and switches characters", async () => {
  const config = await siteConfig({ command: "serve", mode: "test", isPreview: false });
  const server = await createServer({
    ...config,
    root: fileURLToPath(new URL("..", import.meta.url)),
    configFile: false,
    envDir: false,
    logLevel: "silent",
    server: { ...config.server, host: "127.0.0.1", port: 0 },
  });
  await server.listen();
  const browser = await chromium.launch({
    executablePath: executablePath!, headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    page.setDefaultTimeout(60000);
    const models: string[] = [], errors: string[] = [];
    page.on("request", request => { if (request.resourceType() === "fetch" && new URL(request.url()).pathname.endsWith(".clm")) models.push(new URL(request.url()).pathname); });
    page.on("pageerror", error => errors.push(error.message));
    // Nothing named is the live page; `probe` only exposes the editor to this test.
    await page.goto(`${server.resolvedUrls!.local[0]}?demo=live&probe=1`);
    const canvas = page.locator("canvas[data-catchlight-viewport]");
    await canvas.waitFor();
    expect(await page.getByRole("radio", { name: "Q版" }).getAttribute("aria-checked")).toBe("true");
    expect(models).toHaveLength(1);
    expect(models[0]).toContain("dafeiyu-q");

    /** The pose of the session whose file name contains `file`. */
    const read = (file: string, names: string[]) => page.evaluate(async ([file, names]) => {
      const { editor } = (globalThis as unknown as { __catchlightProbe: { editor: import("@catchlight/core").Editor } }).__catchlightProbe;
      const info = (await editor.listSessions()).find(s => JSON.stringify(s).includes(file as string));
      const session = info && editor.session(info.session);
      if (!session) return undefined;
      return Object.fromEntries(session.params().filter(p => (names as string[]).includes(p.name)).map(p => [p.name, session.paramValue(p.id)]));
    }, [file, names] as const);
    const until = async (what: string, check: () => Promise<boolean>) => {
      for (let tries = 0; tries < 80; tries++) { if (await check()) return; await page.waitForTimeout(100); }
      throw new Error(`timed out waiting until ${what}`);
    };

    const box = (await canvas.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.95, box.y + box.height * 0.05, { steps: 5 });
    await until("the head turns up and to the right", async () => {
      const pose = await read("dafeiyu-q", ["head_yaw", "head_pitch"]);
      return (pose?.head_yaw ?? 0) > 15 && (pose?.head_pitch ?? 0) > 8;
    });
    await page.mouse.move(box.x + box.width * 0.05, box.y + box.height * 0.5, { steps: 5 });
    await until("the head turns left", async () => ((await read("dafeiyu-q", ["head_yaw"]))?.head_yaw ?? 0) < -15);

    await page.getByRole("button", { name: "害羞", exact: true }).click();
    await until("she blushes", async () => ((await read("dafeiyu-q", ["face_blush"]))?.face_blush ?? 0) > 0.9);
    expect(await page.getByRole("button", { name: "害羞", exact: true }).getAttribute("aria-pressed")).toBe("true");

    // A tap on the middle of the face is a poke, and she says so.
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.36);
    await page.locator(".live-bubble").waitFor();
    await until("the poke opens her mouth", async () => ((await read("dafeiyu-q", ["mouth_open"]))?.mouth_open ?? 0) > 0.4);

    await page.getByRole("button", { name: "挥手", exact: true }).click();
    await until("an arm goes up", async () => ((await read("dafeiyu-q", ["arm_left_raise"]))?.arm_left_raise ?? 0) > 0.7);

    await page.getByRole("radio", { name: "成年版" }).click();
    await until("the grown-up rig is posed", async () => (await read("dafeiyu-adult", ["breath"])) !== undefined);
    expect(await page.getByRole("radio", { name: "成年版" }).getAttribute("aria-checked")).toBe("true");
    expect(models.filter(path => path.includes("dafeiyu-adult"))).toHaveLength(1);
    await page.mouse.move(box.x + box.width * 0.95, box.y + box.height * 0.3, { steps: 5 });
    await until("she looks right too", async () => ((await read("dafeiyu-adult", ["head_yaw"]))?.head_yaw ?? 0) > 15);
    // Back again is instant: both stay open, and neither is fetched twice.
    await page.getByRole("radio", { name: "Q版" }).click();
    await page.waitForTimeout(300);
    expect(models).toHaveLength(2);
    expect(errors).toEqual([]);
  } finally {
    await browser.close();
    await server.close();
  }
}, 180000);
