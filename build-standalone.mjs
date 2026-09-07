// Builds dist/eyebrow-flappy-bird.html: a single self-contained page with the
// game, MediaPipe's JS bundle and WASM loader, and gzipped+base64 copies of the
// WASM binary and face model embedded inline. It runs without any network
// access, which makes it suitable for hosts with strict content policies.
//
//   npm install
//   npm run build
//
// The face model is downloaded once and cached in .cache/.

import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const modelCache = path.join(root, ".cache", "face_landmarker.task");

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function loadModel() {
  if (process.env.MODEL_PATH) return readFile(process.env.MODEL_PATH);
  if (await exists(modelCache)) return readFile(modelCache);
  console.log("Downloading face model…");
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`Model download failed: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(path.dirname(modelCache), { recursive: true });
  await writeFile(modelCache, buf);
  return buf;
}

function packed(buf) {
  return gzipSync(buf, { level: 9 }).toString("base64");
}

const mpDir = process.env.MP_DIR || path.join(root, "node_modules", "@mediapipe", "tasks-vision");
const mpVersion = JSON.parse(await readFile(path.join(mpDir, "package.json"), "utf8")).version;
const [html, css, game, loaderJs, bundleJs, wasm, model] = await Promise.all([
  readFile(path.join(root, "index.html"), "utf8"),
  readFile(path.join(root, "style.css"), "utf8"),
  readFile(path.join(root, "game.js"), "utf8"),
  readFile(path.join(mpDir, "wasm", "vision_wasm_internal.js"), "utf8"),
  readFile(path.join(mpDir, "vision_bundle.mjs"), "utf8"),
  readFile(path.join(mpDir, "wasm", "vision_wasm_internal.wasm")),
  loadModel(),
]);

// Expose the bundle's exports on window.MP instead of as ES module exports so
// it can live in an inline <script type="module"> and be read by the game.
const exportMatch = bundleJs.match(/export\{([^}]*)\};?\s*(?:\/\/# sourceMappingURL=\S*)?\s*$/);
if (!exportMatch) throw new Error("Could not find the export list at the end of vision_bundle.mjs");
const exportsObj = exportMatch[1]
  .split(",")
  .map((pair) => pair.trim().split(/\s+as\s+/))
  .map(([local, name]) => `${name || local}:${local}`)
  .join(",");
const bundleInline = bundleJs.slice(0, exportMatch.index) + `window.MP={${exportsObj}};`;

for (const [name, text] of [["bundle", bundleInline], ["loader", loaderJs], ["game", game]]) {
  if (/<\/script/i.test(text)) throw new Error(`${name} contains </script>, cannot inline`);
}

const main = html.match(/<main[\s\S]*<\/main>/);
if (!main) throw new Error("index.html has no <main>");
const fontLink = html.match(/<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com[^>]*>/)?.[0] || "";

const out = `<title>Eyebrow Flappy Bird</title>
<meta name="description" content="Flappy Bird you play with your face: raise your eyebrows to flap, blink to duck." />
${fontLink}
<style>
${css}
</style>
${main[0]}
<!-- MediaPipe Tasks Vision ${mpVersion} (Apache-2.0), inlined -->
<script>
${loaderJs}
</script>
<script type="module">
${bundleInline}
</script>
<script type="application/octet-stream" id="mp-wasm">
${packed(wasm)}
</script>
<script type="application/octet-stream" id="mp-model">
${packed(model)}
</script>
<script type="module">
${game}
</script>
`;

const dist = path.join(root, "dist");
await mkdir(dist, { recursive: true });
const outPath = path.join(dist, "eyebrow-flappy-bird.html");
await writeFile(outPath, out);
console.log(`Wrote ${outPath} (${(Buffer.byteLength(out) / 1e6).toFixed(1)} MB)`);
