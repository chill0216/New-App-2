# 🤨 Eyebrow Flappy Bird

Flappy Bird, except the controller is your face.

- **Raise your eyebrows** → the bird flaps. Keep them up to glide.
- **Blink** → the bird ducks (squashes flat and drops faster). Some pipes have a
  gap so narrow you *must* blink to get through. Yes, that means flying blind.

Everything runs in the browser. Your webcam feed never leaves your device: face
tracking uses [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
running locally via WebAssembly/WebGL, and the game reads its eyebrow and blink
blendshape scores.

## Play it

There's no build step. It's three static files (`index.html`, `style.css`,
`game.js`), but the camera API requires a secure context, so serve it over
HTTPS or localhost rather than opening the file directly:

```sh
# any static server works, e.g.
python3 -m http.server 8000
# then open http://localhost:8000
```

The first load downloads the MediaPipe WASM runtime and the face model (about
3.5 MB) from the CDN.

### Standalone single-file build

`npm run build` writes `dist/eyebrow-flappy-bird.html`: one self-contained
page with the game, MediaPipe's runtime, and gzipped copies of the WASM binary
and face model embedded inline (about 8.7 MB). It needs no network access at
all, which makes it work on hosts with strict content security policies
(for example a claude.ai Artifact) and offline. The first run does
`npm install` for the MediaPipe package and downloads the model once into
`.cache/`.

### Deploying

The included GitHub Actions workflow (`.github/workflows/pages.yml`) publishes
the repo root to GitHub Pages on every push to `main`. Enable it once in the
repository settings: **Settings → Pages → Source: GitHub Actions**.

## Controls

| Action | Face                | Keyboard / touch            |
| ------ | ------------------- | --------------------------- |
| Flap   | Raise both eyebrows | `Space`, `↑`, `W`, or tap   |
| Glide  | Hold eyebrows up    | Hold the same key or touch  |
| Duck   | Blink (hold to stay ducked) | Hold `↓` or `S`     |
| Mute   |                     | `M`                         |

Keyboard controls always work, even in face mode, so you can rescue a run.

## Tuning

Open the ⚙️ panel in the corner:

- **Eyebrow sensitivity** – how far above your resting brow position counts as
  a flap. Lower it for subtle raises; raise it if the bird flaps on its own.
  The game also tracks your resting brow level automatically, so a naturally
  "surprised" face won't spam flaps.
- **Blink threshold** – how closed your eyes must be to duck. Lower it if
  ducking feels unresponsive; raise it if you duck accidentally.
- **Show camera preview** – hide the picture-in-picture if it's distracting.
  The live meters stay visible so you can see what the tracker sees.

Settings and your best score are stored in `localStorage`.

## How the face input works

`game.js` pulls three MediaPipe blendshapes per frame:

- `browInnerUp` (and the outer-brow pair as a backup) → **flap** on the rising
  edge, with hysteresis so you have to relax and re-raise for the next flap.
- `eyeBlinkLeft` / `eyeBlinkRight` averaged → **duck** while above the
  threshold.

The bird's own eyebrow and eyelid mirror your face in real time, which is
exactly as dumb as it sounds.

## Browser support

Works in current Chrome, Edge, Firefox, and Safari (desktop and mobile). The
tracker uses the GPU delegate when available and falls back to CPU.
