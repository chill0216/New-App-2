// Eyebrow Flappy Bird
// Raise your eyebrows to flap, blink to duck. Face tracking runs entirely in
// the browser via MediaPipe's Face Landmarker (blendshapes). No video is sent
// anywhere.

const MP_VERSION = "0.10.14";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// ---------------------------------------------------------------------------
// World constants
// ---------------------------------------------------------------------------
const W = 480;
const H = 640;
const GROUND_H = 76;
const SKY_H = H - GROUND_H;

// Tuned for eyebrow cadence (about one raise per second), not finger taps:
// a flap lifts ~80px and takes ~1s to come back down. Holding the brows up
// glides (weak gravity, slow max descent); blinking ducks (heavy gravity).
const GRAVITY = 700;
const FLAP_VY = -340;
const GLIDE_GRAVITY_MULT = 0.3;
const GLIDE_MAX_FALL = 90;
const DUCK_GRAVITY_MULT = 1.35;
const MAX_FALL = 520;

const BIRD_X = 130;
const BIRD_W = 40; // hitbox width
const BIRD_H = 32; // hitbox height (upright)
const BIRD_H_DUCK = 15; // hitbox height (ducking)

const PIPE_W = 72;
const PIPE_SPACING = 275;
const GAP_NORMAL = 185;
const GAP_DUCK = 108;
const FIRST_PIPE_X = W + 140;

const RESTART_DELAY_MS = 750;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const canvas = $("game");
const ctx = canvas.getContext("2d");
const video = $("cam");

const ui = {
  overlayStart: $("overlay-start"),
  overlayDead: $("overlay-dead"),
  loadStatus: $("load-status"),
  btnCamera: $("btn-camera"),
  btnKeys: $("btn-keys"),
  btnOpenTab: $("btn-open-tab"),
  selfUrl: $("self-url"),
  diag: $("diag"),
  btnShare: $("btn-share"),
  btnRetry: $("btn-retry"),
  btnSettings: $("btn-settings"),
  btnCloseSettings: $("btn-close-settings"),
  btnMute: $("btn-mute"),
  settings: $("settings"),
  deadTitle: $("dead-title"),
  deadScore: $("dead-score"),
  deadBest: $("dead-best"),
  deadHint: $("dead-hint"),
  faceHud: $("face-hud"),
  meterBrow: $("meter-brow"),
  meterBlink: $("meter-blink"),
  thrBrow: $("thr-brow"),
  thrBlink: $("thr-blink"),
  faceStatus: $("face-status"),
  rngBrow: $("rng-brow"),
  rngBlink: $("rng-blink"),
  valBrow: $("val-brow"),
  valBlink: $("val-blink"),
  chkCam: $("chk-cam"),
};

// ---------------------------------------------------------------------------
// Persistent settings
// ---------------------------------------------------------------------------
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* ignore */
    }
  },
};

const settings = {
  browSens: store.get("ebfb.browSens", 0.35),
  blinkThr: store.get("ebfb.blinkThr", 0.5),
  showCam: store.get("ebfb.showCam", true),
  muted: store.get("ebfb.muted", false),
};

// ---------------------------------------------------------------------------
// Input (unified across face + keyboard + touch)
// ---------------------------------------------------------------------------
const input = {
  flapQueued: false,
  keyDuck: false,
  faceDuck: false,
  keyLift: false,
  faceLift: false,
  // Continuous levels 0..1 used to animate the bird's face.
  browLevel: 0,
  blinkLevel: 0,
  faceFound: false,
  usingFace: false,
  get duck() {
    return this.keyDuck || this.faceDuck;
  },
  get lift() {
    return this.keyLift || this.faceLift;
  },
};

function queueFlap() {
  input.flapQueued = true;
}

// ---------------------------------------------------------------------------
// Face tracking
// ---------------------------------------------------------------------------
const face = {
  landmarker: null,
  modelPromise: null,
  stream: null,
  lastVideoTime: -1,
  brow: 0,
  browBaseline: 0.08,
  browArmed: true, // true => a raise will trigger a flap
  blink: 0,
  blinking: false,
  lastSeen: 0,
};

// The face engine can come from two places:
//  1. CDN (default for the repo): MediaPipe's JS bundle + WASM + model are
//     fetched at runtime.
//  2. Inline (standalone build, see build-standalone.mjs): the loader script,
//     the JS bundle, and gzipped+base64 copies of the WASM binary and the
//     model are embedded in the page, so it runs with zero network access.
function hasInlineEngine() {
  return !!(
    window.MP &&
    window.ModuleFactory &&
    document.getElementById("mp-wasm") &&
    document.getElementById("mp-model")
  );
}

async function decodeInlineAsset(id) {
  const b64 = document.getElementById(id).textContent.replace(/\s+/g, "");
  const bin = atob(b64);
  const packed = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) packed[i] = bin.charCodeAt(i);
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser can't unpack the bundled face model (no DecompressionStream).");
  }
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const LANDMARKER_OPTIONS = {
  outputFaceBlendshapes: true,
  outputFacialTransformationMatrixes: false,
  runningMode: "VIDEO",
  numFaces: 1,
};

async function createWithFallback(create) {
  try {
    return await create("GPU");
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    return await create("CPU");
  }
}

function loadModel() {
  if (face.modelPromise) return face.modelPromise;
  face.modelPromise = hasInlineEngine() ? loadInlineEngine() : loadCdnEngine();
  return face.modelPromise;
}

async function loadCdnEngine() {
  const { FaceLandmarker, FilesetResolver } = await import(MP_URL);
  const vision = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
  face.landmarker = await createWithFallback((delegate) =>
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      ...LANDMARKER_OPTIONS,
    })
  );
  return face.landmarker;
}

async function loadInlineEngine() {
  setLoadStatus("Unpacking face tracker…");
  const [wasmBytes, modelBytes] = await Promise.all([
    decodeInlineAsset("mp-wasm"),
    decodeInlineAsset("mp-model"),
  ]);
  const { FaceLandmarker } = window.MP;
  // An empty loader path tells MediaPipe to use the already-defined global
  // ModuleFactory instead of injecting a <script>; Module.wasmBinary makes
  // Emscripten instantiate from memory instead of fetching the .wasm.
  const fileset = { wasmLoaderPath: "", wasmBinaryPath: "vision_wasm_internal.wasm" };
  face.landmarker = await createWithFallback((delegate) => {
    window.Module = { wasmBinary: wasmBytes.slice() };
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: modelBytes, delegate },
      ...LANDMARKER_OPTIONS,
    });
  });
  return face.landmarker;
}

function setLoadStatus(msg) {
  if (!ui.overlayStart.hidden) ui.loadStatus.textContent = msg;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  face.stream = stream;
  video.srcObject = stream;
  await video.play();
}

function faceLoop() {
  const lm = face.landmarker;
  if (lm && video.readyState >= 2 && video.currentTime !== face.lastVideoTime) {
    face.lastVideoTime = video.currentTime;
    let result = null;
    try {
      result = lm.detectForVideo(video, performance.now());
    } catch (err) {
      console.warn("detectForVideo failed", err);
    }
    const shapes = result?.faceBlendshapes?.[0]?.categories;
    if (shapes && shapes.length) {
      processBlendshapes(shapes);
      face.lastSeen = performance.now();
      input.faceFound = true;
    } else if (performance.now() - face.lastSeen > 400) {
      input.faceFound = false;
      input.faceDuck = false;
      input.faceLift = false;
      face.blinking = false;
    }
  }
  if ("requestVideoFrameCallback" in video) {
    video.requestVideoFrameCallback(() => faceLoop());
  } else {
    requestAnimationFrame(faceLoop);
  }
}

function processBlendshapes(categories) {
  const score = (name) => {
    for (let i = 0; i < categories.length; i++) {
      if (categories[i].categoryName === name) return categories[i].score;
    }
    return 0;
  };

  // --- Eyebrows -----------------------------------------------------------
  const browRaw = Math.max(
    score("browInnerUp"),
    (score("browOuterUpLeft") + score("browOuterUpRight")) / 2
  );
  face.brow += (browRaw - face.brow) * 0.6;

  // Adaptive resting baseline: follows the brow down quickly and creeps up
  // slowly, so a person's natural resting brow doesn't count as a raise.
  const sens = settings.browSens;
  if (face.brow < face.browBaseline) {
    face.browBaseline += (face.brow - face.browBaseline) * 0.15;
  } else if (face.brow < face.browBaseline + sens * 0.4) {
    face.browBaseline += (face.brow - face.browBaseline) * 0.01;
  }
  face.browBaseline = Math.min(face.browBaseline, 0.45);

  const rise = face.brow - face.browBaseline;
  if (face.browArmed && rise > sens) {
    face.browArmed = false;
    queueFlap();
  } else if (!face.browArmed && rise < sens * 0.45) {
    face.browArmed = true;
  }
  input.faceLift = rise > sens * 0.5;
  input.browLevel = clamp(rise / sens, 0, 1.25);

  // --- Blink --------------------------------------------------------------
  const blinkRaw = (score("eyeBlinkLeft") + score("eyeBlinkRight")) / 2;
  face.blink += (blinkRaw - face.blink) * 0.75;
  const thr = settings.blinkThr;
  if (!face.blinking && face.blink > thr) face.blinking = true;
  else if (face.blinking && face.blink < thr * 0.6) face.blinking = false;
  input.faceDuck = face.blinking;
  input.blinkLevel = clamp(face.blink, 0, 1);
}

function updateFaceHud() {
  if (!input.usingFace) return;
  const browPct = clamp(input.browLevel * 100 * 0.8, 0, 100); // threshold line sits at 80%
  ui.meterBrow.style.width = browPct + "%";
  ui.thrBrow.style.left = "80%";
  ui.meterBlink.style.width = clamp(input.blinkLevel * 100, 0, 100) + "%";
  ui.thrBlink.style.left = clamp(settings.blinkThr * 100, 0, 100) + "%";
  if (input.faceFound) {
    ui.faceStatus.textContent = input.faceDuck ? "ducking" : face.browArmed ? "face ok" : "flap!";
    ui.faceStatus.classList.add("ok");
  } else {
    ui.faceStatus.textContent = "no face";
    ui.faceStatus.classList.remove("ok");
  }
}

// ---------------------------------------------------------------------------
// Audio (tiny synth, no assets)
// ---------------------------------------------------------------------------
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch {
      audioCtx = null;
    }
  }
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
}

function beep({ freq = 440, to = null, dur = 0.08, type = "square", gain = 0.08 }) {
  if (settings.muted || !audioCtx) return;
  const t0 = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  if (to) osc.frequency.exponentialRampToValueAtTime(to, t0 + dur);
  g.gain.setValueAtTime(gain, t0);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(g).connect(audioCtx.destination);
  osc.start(t0);
  osc.stop(t0 + dur + 0.02);
}
const sfx = {
  flap: () => beep({ freq: 520, to: 880, dur: 0.09, type: "square" }),
  duck: () => beep({ freq: 300, to: 160, dur: 0.1, type: "triangle" }),
  score: () => {
    beep({ freq: 880, dur: 0.07, type: "sine", gain: 0.1 });
    setTimeout(() => beep({ freq: 1320, dur: 0.1, type: "sine", gain: 0.1 }), 70);
  },
  die: () => beep({ freq: 400, to: 60, dur: 0.45, type: "sawtooth", gain: 0.12 }),
};

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
const state = {
  mode: "boot", // boot | ready | playing | dead
  score: 0,
  best: store.get("ebfb.best", 0),
  time: 0,
  diedAt: 0,
  flash: 0,
  shake: 0,
  scroll: 0,
  pipes: [],
  clouds: [],
  bird: {
    y: SKY_H / 2,
    vy: 0,
    rot: 0,
    squash: 0, // 0 upright .. 1 fully ducked (animated)
    wing: 0, // wing flap animation timer
    browAnim: 0,
    blinkAnim: 0,
    wasDucking: false,
  },
  lastDuckGate: false,
};

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function rand(lo, hi) {
  return lo + Math.random() * (hi - lo);
}

function resetRound() {
  state.score = 0;
  state.time = 0;
  state.pipes = [];
  state.lastDuckGate = false;
  const b = state.bird;
  b.y = SKY_H / 2;
  b.vy = 0;
  b.rot = 0;
  b.squash = 0;
  b.wing = 0;
  spawnPipe(FIRST_PIPE_X);
}

function initClouds() {
  state.clouds = [];
  for (let i = 0; i < 6; i++) {
    state.clouds.push({
      x: rand(0, W),
      y: rand(30, SKY_H * 0.6),
      s: rand(0.6, 1.3),
      v: rand(10, 22),
    });
  }
}

function speedForScore(score) {
  return 160 + Math.min(score, 40) * 2.5;
}

function spawnPipe(x) {
  let duckGate = false;
  if (state.score >= 2 && !state.lastDuckGate && Math.random() < 0.33) duckGate = true;
  state.lastDuckGate = duckGate;
  const gap = duckGate ? GAP_DUCK : GAP_NORMAL;
  const margin = 60;
  const gapY = rand(margin + gap / 2, SKY_H - margin - gap / 2);
  state.pipes.push({ x, gapY, gap, duckGate, scored: false });
}

function startGame() {
  resetRound();
  state.mode = "playing";
  ui.overlayDead.hidden = true;
  ui.overlayStart.hidden = true;
}

function die() {
  if (state.mode !== "playing") return;
  state.mode = "dead";
  state.diedAt = performance.now();
  state.flash = 1;
  state.shake = 1;
  sfx.die();
  if (state.score > state.best) {
    state.best = state.score;
    store.set("ebfb.best", state.best);
  }
  ui.deadScore.textContent = state.score;
  ui.deadBest.textContent = state.best;
  ui.deadTitle.textContent = pickDeathTitle(state.score);
  ui.deadHint.textContent = input.usingFace
    ? "Raise your eyebrows to go again"
    : "Press Space to go again";
  setTimeout(() => {
    if (state.mode === "dead") ui.overlayDead.hidden = false;
  }, 450);
}

function pickDeathTitle(score) {
  if (score === 0) return "Splat.";
  if (score < 5) return "Brow-tally fine.";
  if (score < 10) return "Eyebrows on fleek.";
  if (score < 20) return "Unibrow unlocked.";
  if (score < 40) return "Forehead of steel.";
  return "Are you even human?";
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------
function update(dt) {
  const b = state.bird;
  state.time += dt;

  // Facial animation on the bird follows the player's face (or keys).
  if (input.usingFace) {
    b.browAnim = lerp(b.browAnim, input.browLevel, 1 - Math.exp(-dt * 18));
  } else {
    // Keyboard: a flap pops the brow up; it stays up while held, then relaxes.
    b.browAnim = lerp(b.browAnim, input.keyLift ? 1 : 0, 1 - Math.exp(-dt * 5));
  }
  const targetBlink = input.usingFace ? input.blinkLevel : input.duck ? 1 : 0;
  b.blinkAnim = lerp(b.blinkAnim, targetBlink, 1 - Math.exp(-dt * 22));
  if (b.wing > 0) b.wing = Math.max(0, b.wing - dt * 6);
  state.flash = Math.max(0, state.flash - dt * 3);
  state.shake = Math.max(0, state.shake - dt * 2.5);

  for (const c of state.clouds) {
    c.x -= c.v * dt * (state.mode === "playing" ? 1 : 0.4);
    if (c.x < -140) {
      c.x = W + 60;
      c.y = rand(30, SKY_H * 0.6);
    }
  }

  if (state.mode === "ready" || state.mode === "dead") {
    if (state.mode === "ready") {
      b.y = SKY_H / 2 + Math.sin(state.time * 3) * 10;
      b.rot = Math.sin(state.time * 3) * 0.08;
      state.scroll += 60 * dt;
      if (input.flapQueued) {
        input.flapQueued = false;
        if (!input.usingFace) b.browAnim = 1.2;
        startGame();
        flap();
      }
    } else {
      // Dead: bird drops to the ground.
      b.vy = Math.min(MAX_FALL, b.vy + GRAVITY * dt);
      b.y += b.vy * dt;
      const floor = SKY_H - BIRD_H / 2;
      if (b.y > floor) {
        b.y = floor;
        b.vy = 0;
      }
      b.rot = lerp(b.rot, Math.PI / 2, 1 - Math.exp(-dt * 6));
      if (input.flapQueued) {
        input.flapQueued = false;
        if (performance.now() - state.diedAt > RESTART_DELAY_MS) {
          if (!input.usingFace) b.browAnim = 1.2;
          startGame();
          flap();
        }
      }
    }
    return;
  }

  if (state.mode !== "playing") return;

  // ---- Bird physics ----
  const ducking = input.duck;
  if (ducking && !b.wasDucking) sfx.duck();
  b.wasDucking = ducking;
  b.squash = lerp(b.squash, ducking ? 1 : 0, 1 - Math.exp(-dt * 16));

  if (input.flapQueued) {
    input.flapQueued = false;
    flap();
  }
  // Brows up = glide (works while ducking too); blink = duck (a bit heavier).
  const gliding = input.lift;
  const g = GRAVITY * (gliding ? GLIDE_GRAVITY_MULT : 1) * (ducking ? DUCK_GRAVITY_MULT : 1);
  b.vy = Math.min(gliding ? GLIDE_MAX_FALL : MAX_FALL, b.vy + g * dt);
  b.y += b.vy * dt;
  if (b.y < 8) {
    b.y = 8;
    b.vy = Math.max(0, b.vy);
  }
  const targetRot = clamp(b.vy / 600, -0.5, 1.2);
  b.rot = lerp(b.rot, targetRot, 1 - Math.exp(-dt * 8));

  // ---- Pipes ----
  const speed = speedForScore(state.score);
  state.scroll += speed * dt;
  for (const p of state.pipes) p.x -= speed * dt;
  const last = state.pipes[state.pipes.length - 1];
  if (!last || last.x < W - PIPE_SPACING) spawnPipe((last ? last.x : W) + PIPE_SPACING);
  state.pipes = state.pipes.filter((p) => p.x > -PIPE_W - 10);

  // ---- Scoring + collision ----
  const hitH = lerp(BIRD_H, BIRD_H_DUCK, b.squash);
  const bx0 = BIRD_X - BIRD_W / 2 + 4;
  const bx1 = BIRD_X + BIRD_W / 2 - 4;
  const by0 = b.y - hitH / 2;
  const by1 = b.y + hitH / 2;

  if (by1 >= SKY_H) {
    b.y = SKY_H - hitH / 2;
    die();
    return;
  }

  for (const p of state.pipes) {
    if (!p.scored && p.x + PIPE_W < BIRD_X) {
      p.scored = true;
      state.score++;
      sfx.score();
    }
    const px0 = p.x;
    const px1 = p.x + PIPE_W;
    if (bx1 > px0 && bx0 < px1) {
      const gapTop = p.gapY - p.gap / 2;
      const gapBot = p.gapY + p.gap / 2;
      if (by0 < gapTop || by1 > gapBot) {
        die();
        return;
      }
    }
  }
}

function flap() {
  const b = state.bird;
  b.vy = FLAP_VY;
  b.wing = 1;
  if (!input.usingFace) b.browAnim = 1.2;
  sfx.flap();
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function draw() {
  ctx.save();
  if (state.shake > 0) {
    const s = state.shake * 6;
    ctx.translate(rand(-s, s), rand(-s, s));
  }

  drawSky();
  drawClouds();
  drawPipes();
  drawGround();
  drawBird();
  drawScore();

  ctx.restore();

  if (state.flash > 0) {
    ctx.fillStyle = `rgba(255,255,255,${state.flash * 0.8})`;
    ctx.fillRect(0, 0, W, H);
  }
}

function drawSky() {
  const grd = ctx.createLinearGradient(0, 0, 0, SKY_H);
  grd.addColorStop(0, "#5ec0ff");
  grd.addColorStop(1, "#bdeaff");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, W, SKY_H);

  // Distant skyline.
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  const off = (state.scroll * 0.15) % 90;
  for (let x = -off - 90; x < W + 90; x += 90) {
    const h = 40 + ((Math.floor((x + off) / 90) * 37) % 50);
    ctx.fillRect(x, SKY_H - h, 46, h);
    ctx.fillRect(x + 54, SKY_H - h * 0.6, 26, h * 0.6);
  }
}

function drawClouds() {
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  for (const c of state.clouds) {
    ctx.beginPath();
    ctx.ellipse(c.x, c.y, 34 * c.s, 14 * c.s, 0, 0, Math.PI * 2);
    ctx.ellipse(c.x - 20 * c.s, c.y + 4 * c.s, 20 * c.s, 11 * c.s, 0, 0, Math.PI * 2);
    ctx.ellipse(c.x + 22 * c.s, c.y + 4 * c.s, 22 * c.s, 12 * c.s, 0, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawPipes() {
  for (const p of state.pipes) {
    const gapTop = p.gapY - p.gap / 2;
    const gapBot = p.gapY + p.gap / 2;
    drawPipeSegment(p.x, 0, gapTop, true, p.duckGate);
    drawPipeSegment(p.x, gapBot, SKY_H - gapBot, false, p.duckGate);

    if (p.duckGate) {
      // Sign: BLINK!
      const sx = p.x + PIPE_W / 2;
      const sy = gapTop - 34;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(Math.sin(state.time * 6 + p.x) * 0.06);
      ctx.fillStyle = "#ff5c8a";
      roundRect(-34, -13, 68, 26, 8);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "900 15px " + fontStack();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("BLINK!", 0, 1);
      ctx.restore();
    }
  }
}

function drawPipeSegment(x, y, h, isTop, hazard) {
  if (h <= 0) return;
  const body = hazard ? "#e0567a" : "#5ccf5c";
  const dark = hazard ? "#a8385a" : "#3aa03a";
  const light = hazard ? "#ff9cb8" : "#9cf59c";
  const capH = 26;

  ctx.fillStyle = body;
  ctx.fillRect(x + 4, y, PIPE_W - 8, h);
  ctx.fillStyle = light;
  ctx.fillRect(x + 8, y, 10, h);
  ctx.fillStyle = dark;
  ctx.fillRect(x + PIPE_W - 16, y, 8, h);

  const capY = isTop ? y + h - capH : y;
  ctx.fillStyle = body;
  roundRect(x, capY, PIPE_W, capH, 5);
  ctx.fill();
  ctx.strokeStyle = dark;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = light;
  ctx.fillRect(x + 6, capY + 4, 12, capH - 8);

  if (hazard) {
    // Hazard stripes on the caps.
    ctx.save();
    ctx.beginPath();
    roundRect(x, capY, PIPE_W, capH, 5);
    ctx.clip();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    for (let sx = -capH; sx < PIPE_W + capH; sx += 18) {
      ctx.beginPath();
      ctx.moveTo(x + sx, capY);
      ctx.lineTo(x + sx + 8, capY);
      ctx.lineTo(x + sx + 8 - capH, capY + capH);
      ctx.lineTo(x + sx - capH, capY + capH);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }
}

function drawGround() {
  ctx.fillStyle = "#d9b26a";
  ctx.fillRect(0, SKY_H, W, GROUND_H);
  ctx.fillStyle = "#7ed957";
  ctx.fillRect(0, SKY_H, W, 14);
  ctx.fillStyle = "#5fb843";
  const off = state.scroll % 28;
  for (let x = -off; x < W; x += 28) {
    ctx.fillRect(x, SKY_H + 8, 14, 6);
  }
  ctx.fillStyle = "#c39a55";
  const off2 = (state.scroll * 0.7) % 60;
  for (let x = -off2; x < W; x += 60) {
    ctx.fillRect(x, SKY_H + 34, 30, 6);
    ctx.fillRect(x + 20, SKY_H + 54, 22, 6);
  }
}

function drawBird() {
  const b = state.bird;
  ctx.save();
  ctx.translate(BIRD_X, b.y);
  ctx.rotate(b.rot);
  const sq = b.squash;
  ctx.scale(1 + sq * 0.28, 1 - sq * 0.55);

  // Tail feathers
  ctx.fillStyle = "#ffb02e";
  ctx.beginPath();
  ctx.moveTo(-18, -2);
  ctx.lineTo(-34, -10);
  ctx.lineTo(-30, 2);
  ctx.lineTo(-34, 12);
  ctx.closePath();
  ctx.fill();

  // Body
  ctx.fillStyle = "#ffd23f";
  ctx.strokeStyle = "#5a3a00";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.ellipse(0, 0, 22, 18, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Belly
  ctx.fillStyle = "#fff0a8";
  ctx.beginPath();
  ctx.ellipse(-3, 6, 12, 8, 0, 0, Math.PI * 2);
  ctx.fill();

  // Wing (flaps on input)
  const wingAngle = -0.9 * Math.sin(Math.min(1, b.wing) * Math.PI) - 0.15;
  ctx.save();
  ctx.translate(-6, 2);
  ctx.rotate(wingAngle);
  ctx.fillStyle = "#ffb02e";
  ctx.beginPath();
  ctx.ellipse(-4, 4, 14, 8, 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // Beak
  ctx.fillStyle = "#ff7a2e";
  ctx.beginPath();
  ctx.moveTo(16, -2);
  ctx.lineTo(34, 3);
  ctx.lineTo(16, 9);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Eye
  const ex = 9;
  const ey = -6;
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(ex, ey, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#222";
  ctx.beginPath();
  ctx.arc(ex + 2.5, ey - 0.5, 3.6, 0, Math.PI * 2);
  ctx.fill();

  // Eyelid closes with blink level
  const lid = clamp(b.blinkAnim, 0, 1);
  if (lid > 0.02) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(ex, ey, 8.5, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "#ffd23f";
    ctx.fillRect(ex - 10, ey - 10, 20, 20 * lid);
    ctx.restore();
    if (lid > 0.85) {
      ctx.strokeStyle = "#5a3a00";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(ex - 7, ey + 1);
      ctx.quadraticCurveTo(ex, ey + 4, ex + 7, ey + 1);
      ctx.stroke();
    }
  }

  // The star of the show: the eyebrow.
  const raise = clamp(b.browAnim, 0, 1.25);
  const byOff = -10 - raise * 13;
  ctx.strokeStyle = "#3b2400";
  ctx.lineCap = "round";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(ex - 9, ey + byOff + 3 - raise * 2);
  ctx.quadraticCurveTo(ex, ey + byOff - 4 - raise * 3, ex + 9, ey + byOff + 2);
  ctx.stroke();

  ctx.restore();
}

function drawScore() {
  if (state.mode === "ready" || state.mode === "boot") return;
  ctx.save();
  ctx.font = "900 56px " + fontStack();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.lineWidth = 8;
  ctx.strokeStyle = "rgba(40,20,0,0.85)";
  ctx.fillStyle = "#fff";
  ctx.strokeText(state.score, W / 2, 26);
  ctx.fillText(state.score, W / 2, 26);
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function fontStack() {
  return '"Nunito","Avenir Next","Segoe UI",system-ui,sans-serif';
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let lastT = 0;
function frame(t) {
  const dt = Math.min(0.033, (t - lastT) / 1000 || 0.016);
  lastT = t;
  update(dt);
  draw();
  updateFaceHud();
  requestAnimationFrame(frame);
}

function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function bindControls() {
  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    if (e.code === "Space" || e.code === "ArrowUp" || e.code === "KeyW") {
      e.preventDefault();
      ensureAudio();
      input.keyLift = true;
      queueFlap();
    } else if (e.code === "ArrowDown" || e.code === "KeyS") {
      e.preventDefault();
      input.keyDuck = true;
    } else if (e.code === "KeyM") {
      toggleMute();
    }
  });
  window.addEventListener("keyup", (e) => {
    if (e.code === "ArrowDown" || e.code === "KeyS") input.keyDuck = false;
    if (e.code === "Space" || e.code === "ArrowUp" || e.code === "KeyW") input.keyLift = false;
  });

  // Tap / click on the canvas flaps (handy on phones when your face is busy).
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    ensureAudio();
    input.keyLift = true;
    queueFlap();
  });
  for (const ev of ["pointerup", "pointercancel", "pointerleave"]) {
    canvas.addEventListener(ev, () => {
      input.keyLift = false;
    });
  }

  ui.btnCamera.addEventListener("click", async () => {
    ensureAudio();
    ui.btnCamera.disabled = true;
    ui.btnKeys.disabled = true;
    ui.loadStatus.textContent = "Asking for camera…";
    try {
      await startCamera();
      ui.loadStatus.textContent = "Warming up face tracking…";
      await loadModel();
      input.usingFace = true;
      ui.faceHud.hidden = false;
      ui.faceHud.classList.toggle("no-preview", !settings.showCam);
      faceLoop();
      ui.overlayStart.hidden = true;
      state.mode = "ready";
      showToast("Raise eyebrows: flap · hold up: glide · blink: duck");
    } catch (err) {
      console.error(err);
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      ui.loadStatus.textContent = denied
        ? isEmbedded()
          ? "This embedded view blocks the camera. Tap \"Open in its own tab\" below, then allow the camera there."
          : "Camera blocked. Allow camera access in your browser settings, or play with the keyboard."
        : "Couldn't start face tracking (" + (err?.message || err) + "). Keyboard still works.";
      if (isEmbedded()) {
        ui.btnOpenTab.hidden = false;
        showTopLevelLinks(topLevelUrlCandidates());
      }
      showDiagnostics(err);
      ui.btnCamera.disabled = false;
      ui.btnKeys.disabled = false;
    }
  });

  // When the page is framed by a host that doesn't grant camera access, the
  // page's own URL opened as a top-level tab usually does.
  if (isEmbedded()) ui.btnOpenTab.hidden = false;
  ui.btnOpenTab.addEventListener("click", () => {
    const urls = topLevelUrlCandidates();
    let win = null;
    try {
      win = window.open(urls[0], "_blank", "noopener");
    } catch {
      win = null;
    }
    showTopLevelLinks(urls);
    if (!win) showToast("Pop-ups are blocked here. Tap a link below instead.");
    showDiagnostics();
  });

  ui.btnKeys.addEventListener("click", () => {
    ensureAudio();
    input.usingFace = false;
    ui.overlayStart.hidden = true;
    state.mode = "ready";
    showToast("Space: flap · hold Space: glide · hold ↓: duck");
  });

  ui.btnRetry.addEventListener("click", () => {
    ensureAudio();
    startGame();
  });

  ui.btnShare.addEventListener("click", shareScore);

  ui.btnSettings.addEventListener("click", () => {
    ui.settings.hidden = !ui.settings.hidden;
  });
  ui.btnCloseSettings.addEventListener("click", () => {
    ui.settings.hidden = true;
  });

  ui.rngBrow.value = settings.browSens;
  ui.valBrow.textContent = settings.browSens.toFixed(2);
  ui.rngBrow.addEventListener("input", () => {
    settings.browSens = parseFloat(ui.rngBrow.value);
    ui.valBrow.textContent = settings.browSens.toFixed(2);
    store.set("ebfb.browSens", settings.browSens);
  });

  ui.rngBlink.value = settings.blinkThr;
  ui.valBlink.textContent = settings.blinkThr.toFixed(2);
  ui.rngBlink.addEventListener("input", () => {
    settings.blinkThr = parseFloat(ui.rngBlink.value);
    ui.valBlink.textContent = settings.blinkThr.toFixed(2);
    store.set("ebfb.blinkThr", settings.blinkThr);
  });

  ui.chkCam.checked = settings.showCam;
  ui.chkCam.addEventListener("change", () => {
    settings.showCam = ui.chkCam.checked;
    store.set("ebfb.showCam", settings.showCam);
    ui.faceHud.classList.toggle("no-preview", !settings.showCam);
  });

  ui.btnMute.addEventListener("click", toggleMute);
  renderMute();

  // Pause face-driven flaps while the tab is hidden so returning doesn't
  // instantly fire a queued flap.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) input.flapQueued = false;
  });
}

// Some hosts serve framed pages from "<id>.frame.<host>" and the same page
// top-level from "<id>-top.frame.<host>". Offer that variant first.
function topLevelUrlCandidates() {
  const here = location.href;
  const m = location.hostname.match(/^([^.]+)\.frame\.(.+)$/);
  const urls = [];
  if (m && !m[1].endsWith("-top")) {
    urls.push(`${location.protocol}//${m[1]}-top.frame.${m[2]}${location.pathname}${location.search}`);
  }
  urls.push(here);
  return urls;
}

function showTopLevelLinks(urls) {
  ui.selfUrl.textContent = "";
  const label = document.createElement("span");
  label.textContent = "Open one of these in Safari or Chrome, then allow the camera:";
  ui.selfUrl.appendChild(label);
  for (const url of urls) {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = url;
    ui.selfUrl.appendChild(a);
  }
  ui.selfUrl.hidden = false;
}

async function showDiagnostics(err) {
  const bits = [
    `host=${location.hostname}`,
    `framed=${isEmbedded()}`,
    `secure=${window.isSecureContext}`,
    `mediaDevices=${!!navigator.mediaDevices}`,
  ];
  if (err) bits.push(`err=${err.name || "?"}:${(err.message || "").slice(0, 60)}`);
  try {
    if (navigator.permissions?.query) {
      const st = await navigator.permissions.query({ name: "camera" });
      bits.push(`permission=${st.state}`);
    }
  } catch {
    /* Safari may not support querying camera permission */
  }
  try {
    const pp = document.permissionsPolicy || document.featurePolicy;
    if (pp?.allowsFeature) bits.push(`policyAllowsCamera=${pp.allowsFeature("camera")}`);
  } catch {
    /* ignore */
  }
  ui.diag.textContent = bits.join(" · ");
  ui.diag.hidden = false;
}

function isEmbedded() {
  try {
    return window.top !== window.self;
  } catch {
    return true;
  }
}

function toggleMute() {
  settings.muted = !settings.muted;
  store.set("ebfb.muted", settings.muted);
  renderMute();
}
function renderMute() {
  ui.btnMute.textContent = settings.muted ? "🔇" : "🔊";
}

async function shareScore() {
  const url = location.href.split("#")[0];
  const text = `I scored ${state.score} in Eyebrow Flappy Bird 🤨🐦 — flapping with my eyebrows and ducking by blinking. Beat me:`;
  try {
    if (navigator.share) {
      await navigator.share({ title: "Eyebrow Flappy Bird", text, url });
      return;
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
  }
  try {
    await navigator.clipboard.writeText(`${text} ${url}`);
    showToast("Copied to clipboard 📋");
  } catch {
    showToast("Couldn't share — screenshot it!");
  }
}

let toastEl = null;
let toastTimer = 0;
function showToast(msg) {
  if (!toastEl) {
    toastEl = document.createElement("div");
    Object.assign(toastEl.style, {
      position: "absolute",
      left: "50%",
      top: "14%",
      transform: "translateX(-50%)",
      background: "rgba(18,20,40,0.9)",
      color: "#fff",
      padding: "10px 16px",
      borderRadius: "999px",
      fontWeight: "800",
      fontSize: "15px",
      whiteSpace: "nowrap",
      zIndex: "6",
      pointerEvents: "none",
      transition: "opacity 0.3s",
    });
    $("stage").appendChild(toastEl);
  }
  toastEl.textContent = msg;
  toastEl.style.opacity = "1";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toastEl.style.opacity = "0"), 2200);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  fitCanvas();
  window.addEventListener("resize", fitCanvas);
  initClouds();
  resetRound();
  bindControls();
  requestAnimationFrame(frame);

  const secure = window.isSecureContext && navigator.mediaDevices?.getUserMedia;
  if (!secure) {
    ui.loadStatus.textContent =
      "Camera needs HTTPS (or localhost). Keyboard mode is available.";
    return;
  }

  // Kick off the model download right away so it's ready by the time the
  // player has granted camera access.
  loadModel()
    .then(() => {
      setLoadStatus("Face tracker ready. Your camera stays on your device.");
      ui.btnCamera.disabled = false;
    })
    .catch((err) => {
      console.error(err);
      setLoadStatus(
        hasInlineEngine()
          ? "Couldn't start the face tracker (" + (err?.message || err) + "). Keyboard mode still works."
          : "Couldn't load the face model (offline?). Keyboard mode still works."
      );
      ui.btnCamera.disabled = true;
    });
}

boot();

// Exposed for debugging/automation only.
window.__ebfb = { state, input, settings, queueFlap };
