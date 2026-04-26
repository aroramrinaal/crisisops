// ============================================================
// CrisisOps · scripted playback for cascading_crisis (seed=42)
// purely cosmetic, no backend
// ============================================================

// ---------------- static config ----------------

const ZONES = {
  riverside:      { name: "RIVERSIDE",      tag: "GR-04A", x: 18, y: 78, glyph: "≈" },
  "old-market":   { name: "OLD MARKET",     tag: "GR-08C", x: 36, y: 52, glyph: "▦" },
  "north-hospital":{name: "N. HOSPITAL",    tag: "GR-12B", x: 52, y: 22, glyph: "✚" },
  "rail-yard":    { name: "RAIL YARD",      tag: "GR-15D", x: 70, y: 56, glyph: "▤" },
  "hill-sector":  { name: "HILL SECTOR",    tag: "GR-19E", x: 84, y: 26, glyph: "△" },
};

// percent-of-map positions for unit dots (start = depot, deployed = zone)
const DEPOT = { x: 8, y: 92 };

const UNITS = [
  { id: "rt-1", label: "RT-1", kind: "rescue_team",  short: "RT" },
  { id: "rt-2", label: "RT-2", kind: "rescue_team",  short: "RT" },
  { id: "mu-1", label: "MU-1", kind: "medical",      short: "MU" },
  { id: "st-1", label: "ST-1", kind: "supply",       short: "ST" },
  { id: "eb-1", label: "EB-1", kind: "evac_bus",     short: "EB" },
  { id: "rd-1", label: "RD-1", kind: "recon_drone",  short: "RD" },
];

const REPORTS = {
  "r-1":  { incident: "flood",      glyph: "≈", zone: "riverside",      sev: 4, source: "citizen",  truthy: true  },
  "r-2":  { incident: "fire",       glyph: "▲", zone: "old-market",     sev: 3, source: "sensor",   truthy: true  },
  "r-3":  { incident: "medical",    glyph: "✚", zone: "north-hospital", sev: 5, source: "official", truthy: true  },
  "r-4":  { incident: "chemical",   glyph: "☣", zone: "rail-yard",      sev: 2, source: "unknown",  truthy: false },
  "r-5":  { incident: "landslide",  glyph: "△", zone: "hill-sector",    sev: 4, source: "citizen",  truthy: true  },
  "r-6":  { incident: "fire",       glyph: "▲", zone: "old-market",     sev: 4, source: "citizen",  truthy: true  },
  "r-7":  { incident: "flood",      glyph: "≈", zone: "rail-yard",      sev: 3, source: "social",   truthy: false },
  "r-8":  { incident: "medical",    glyph: "✚", zone: "north-hospital", sev: 3, source: "sensor",   truthy: true  },
  "r-9":  { incident: "structural", glyph: "▣", zone: "hill-sector",    sev: 4, source: "official", truthy: true  },
  "r-10": { incident: "looting",    glyph: "✦", zone: "riverside",      sev: 1, source: "social",   truthy: false },
};

// inter-zone routes (drawn as svg lines)
const ROUTES = [
  ["riverside",      "old-market"],
  ["old-market",     "north-hospital"],
  ["old-market",     "rail-yard"],
  ["north-hospital", "hill-sector"],
  ["rail-yard",      "hill-sector"],
];

// scripted episode (cascading_crisis · hard · seed=42)
const STEPS = [
  { t: "report",   id: "r-1" },
  { t: "report",   id: "r-2" },
  { t: "verify",   reportId: "r-1", reward: 0.05 },
  { t: "allocate", unitId: "rt-1", zone: "riverside",      reward: 0.30, star: true },
  { t: "report",   id: "r-3" },
  { t: "verify",   reportId: "r-2", reward: 0.05 },
  { t: "allocate", unitId: "mu-1", zone: "north-hospital", reward: 0.30, star: true },
  { t: "report",   id: "r-4" },
  { t: "event",    kind: "aftershock", zone: "rail-yard",  banner: "AFTERSHOCK · M5.2", message: "AFTERSHOCK", target: "zone-rail-yard" },
  { t: "report",   id: "r-5" },
  { t: "flag",     reportId: "r-4", reward: 0.05 },
  { t: "allocate", unitId: "st-1", zone: "old-market",     reward: 0.20, star: true },
  { t: "report",   id: "r-6" },
  { t: "event",    kind: "road_closure", route: ["old-market", "rail-yard"], banner: "ROAD CLOSED · OM↔RY", message: "ROAD_CLOSURE" },
  { t: "allocate", unitId: "eb-1", zone: "hill-sector",    reward: 0.25, star: true },
  { t: "verify",   reportId: "r-5", reward: 0.05 },
  { t: "report",   id: "r-7" },
  { t: "report",   id: "r-8" },
  { t: "allocate", unitId: "rd-1", zone: "rail-yard",      reward: 0.10 },
  { t: "flag",     reportId: "r-7", reward: 0.05 },
  { t: "report",   id: "r-9" },
  { t: "report",   id: "r-10" },
  { t: "allocate", unitId: "rt-2", zone: "hill-sector",    reward: 0.20, star: true },
  { t: "flag",     reportId: "r-10", reward: 0.05 },
  { t: "sitrep",   reward: 0.15 },
  { t: "complete", score: 0.743 },
];

// next-action preview text
function nextActionPreview(step) {
  if (!step) return null;
  switch (step.t) {
    case "report":
      return { verb: "await_report", detail: `incoming: ${step.id}`, meta: ["source: stream", "confidence: —"] };
    case "verify":
      return { verb: "verify_report", detail: `target: ${step.reportId}`, meta: ["priority: medium", "confidence: deterministic"] };
    case "flag":
      return { verb: "flag_false_alarm", detail: `target: ${step.reportId}`, meta: ["priority: low", "confidence: deterministic"] };
    case "allocate": {
      const u = UNITS.find(x => x.id === step.unitId);
      const z = ZONES[step.zone];
      return { verb: "allocate_unit", detail: `${u.label} (${u.kind}) → ${z.name}`, meta: ["priority: high", "confidence: deterministic"] };
    }
    case "event":
      return { verb: "EVENT_PENDING", detail: step.banner, meta: ["priority: critical", "confidence: incoming"] };
    case "sitrep":
      return { verb: "publish_sitrep", detail: "broadcast → all zones", meta: ["priority: high", "confidence: deterministic"] };
    case "complete":
      return { verb: "episode_complete", detail: `final score ${step.score.toFixed(3)}`, meta: ["priority: —", "confidence: final"] };
  }
  return null;
}

// ---------------- runtime state ----------------

const state = {
  i: 0,
  score: 0,
  playing: true,
  timer: null,
  speedMs: 1700,
};

const els = {};

// ---------------- bootstrap ----------------

function init() {
  els.shell        = document.querySelector(".shell");
  els.reportsList  = document.getElementById("reports-list");
  els.unitsGrid    = document.getElementById("units-grid");
  els.logBody      = document.getElementById("log-body");
  els.naVerb       = document.getElementById("na-verb");
  els.naDetail     = document.getElementById("na-detail");
  els.naMeta       = document.getElementById("na-meta");
  els.scoreVal     = document.getElementById("score-val");
  els.scoreFill    = document.getElementById("score-fill");
  els.stepCounter  = document.getElementById("step-counter");
  els.btnPlay      = document.getElementById("btn-play");
  els.btnReset     = document.getElementById("btn-reset");
  els.btnStep      = document.getElementById("btn-step");
  els.mapPanel     = document.querySelector(".map");
  els.terrain      = document.getElementById("terrain");

  buildUnits();
  buildZones();
  buildUnitDots();
  drawRoutes();
  buildReportsCount();
  refreshNextAction();
  updateStepCounter();

  els.btnPlay.addEventListener("click", togglePlay);
  els.btnReset.addEventListener("click", reset);
  els.btnStep.addEventListener("click", () => { state.playing = false; setPlayBtn(); advance(); });

  if (state.playing) startTimer();
}

// ---------------- DOM scaffolding ----------------

function buildUnits() {
  els.unitsGrid.innerHTML = "";
  UNITS.forEach(u => {
    const chip = document.createElement("div");
    chip.className = "unit-chip";
    chip.dataset.unit = u.id;
    chip.dataset.state = "idle";
    chip.innerHTML = `
      <div class="unit-icon">${u.short}</div>
      <div class="unit-name">${u.label}</div>
    `;
    els.unitsGrid.appendChild(chip);
  });
}

function buildZones() {
  Object.entries(ZONES).forEach(([id, z]) => {
    const el = document.createElement("div");
    el.className = "zone";
    el.id = `zone-${id}`;
    el.dataset.state = "normal";
    el.dataset.severity = "0";
    el.style.left = `${z.x}%`;
    el.style.top  = `${z.y}%`;
    el.innerHTML = `
      <div class="zone-label">${z.name}</div>
      <div class="zone-box"></div>
      <div class="zone-glyph">${z.glyph}</div>
      <div class="zone-sev" id="sev-${id}">SEV —</div>
      <div class="zone-tag">${z.tag}</div>
    `;
    els.mapPanel.appendChild(el);
  });
}

function buildUnitDots() {
  UNITS.forEach((u, idx) => {
    const dot = document.createElement("div");
    dot.className = "unit-dot";
    dot.id = `dot-${u.id}`;
    dot.dataset.label = u.label;
    // splay depot starting positions slightly so they don't overlap
    const dx = (idx % 3) * 2.2;
    const dy = Math.floor(idx / 3) * 3;
    dot.style.left = `${DEPOT.x + dx}%`;
    dot.style.top  = `${DEPOT.y - dy}%`;
    els.mapPanel.appendChild(dot);
  });
}

function drawRoutes() {
  const svgNS = "http://www.w3.org/2000/svg";
  const overlay = document.getElementById("routes");
  ROUTES.forEach(([a, b]) => {
    const za = ZONES[a], zb = ZONES[b];
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", za.x); line.setAttribute("y1", za.y);
    line.setAttribute("x2", zb.x); line.setAttribute("y2", zb.y);
    line.setAttribute("class", "route");
    line.setAttribute("data-route", `${a}__${b}`);
    line.dataset.state = "open";
    overlay.appendChild(line);
  });
}

function buildReportsCount() {
  document.getElementById("reports-count").textContent = `0 / ${Object.keys(REPORTS).length}`;
}

// ---------------- step engine ----------------

function startTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(advance, state.speedMs);
}

function togglePlay() {
  state.playing = !state.playing;
  setPlayBtn();
  if (state.playing) startTimer();
  else if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

function setPlayBtn() {
  els.btnPlay.classList.toggle("active", state.playing);
  els.btnPlay.textContent = state.playing ? "⏸ PAUSE" : "▶ AUTO-PLAY";
}

function reset() {
  if (state.timer) clearInterval(state.timer);
  state.i = 0;
  state.score = 0;
  state.playing = true;
  // wipe ui
  els.reportsList.innerHTML = "";
  els.logBody.innerHTML = "";
  buildReportsCount();
  Object.keys(ZONES).forEach(id => {
    const z = document.getElementById(`zone-${id}`);
    z.dataset.state = "normal";
    z.dataset.severity = "0";
    document.getElementById(`sev-${id}`).textContent = "SEV —";
  });
  document.querySelectorAll(".route").forEach(r => r.dataset.state = "open");
  document.querySelectorAll(".unit-chip").forEach(c => c.dataset.state = "idle");
  UNITS.forEach((u, idx) => {
    const dot = document.getElementById(`dot-${u.id}`);
    const dx = (idx % 3) * 2.2;
    const dy = Math.floor(idx / 3) * 3;
    dot.style.left = `${DEPOT.x + dx}%`;
    dot.style.top  = `${DEPOT.y - dy}%`;
  });
  updateScore(0);
  refreshNextAction();
  updateStepCounter();
  setPlayBtn();
  startTimer();
}

function advance() {
  if (state.i >= STEPS.length) {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.playing = false;
    setPlayBtn();
    return;
  }
  apply(STEPS[state.i]);
  state.i++;
  updateStepCounter();
  refreshNextAction();
}

function apply(step) {
  switch (step.t) {
    case "report":   onReport(step); break;
    case "verify":   onVerify(step); break;
    case "flag":     onFlag(step);   break;
    case "allocate": onAllocate(step); break;
    case "event":    onEvent(step);  break;
    case "sitrep":   onSitrep(step); break;
    case "complete": onComplete(step); break;
  }
}

// ---------------- step handlers ----------------

function onReport(step) {
  const r = REPORTS[step.id];
  // card
  const card = document.createElement("div");
  card.className = "report-card";
  card.id = `card-${step.id}`;
  card.dataset.state = "unverified";
  const sevCells = Array.from({ length: 5 }, (_, i) =>
    `<div class="cell ${i < r.sev ? "on" : ""}"></div>`
  ).join("");
  card.innerHTML = `
    <div class="rc-row">
      <div class="rc-incident"><span class="glyph">${r.glyph}</span>${r.incident.toUpperCase()}</div>
      <div class="rc-id">${step.id}</div>
    </div>
    <div class="rc-meta">${ZONES[r.zone].name.toLowerCase()} · ${r.source}</div>
    <div class="rc-foot">
      <div style="display:flex;align-items:center;">
        <div class="sev-bar">${sevCells}</div>
        <span class="sev-label">SEV ${r.sev}</span>
      </div>
      <span class="pill" id="pill-${step.id}">UNVERIFIED</span>
    </div>
  `;
  // newest on top
  els.reportsList.prepend(card);

  // bump zone severity if higher
  const zoneEl = document.getElementById(`zone-${r.zone}`);
  const cur = parseInt(zoneEl.dataset.severity || "0", 10);
  if (r.sev > cur && zoneEl.dataset.state !== "blocked") {
    zoneEl.dataset.severity = String(r.sev);
    document.getElementById(`sev-${r.zone}`).textContent = `SEV ${r.sev}`;
  }

  log({ verb: `report_in     ${step.id.padEnd(8)} ${r.zone}`, delta: "—" });
  bumpReportsCount();
}

function onVerify(step) {
  const card = document.getElementById(`card-${step.reportId}`);
  if (card) {
    card.dataset.state = "verified";
    document.getElementById(`pill-${step.reportId}`).textContent = "VERIFIED ✓";
  }
  addScore(step.reward);
  log({ verb: `verify_report ${step.reportId.padEnd(8)}`, delta: `+${step.reward.toFixed(2)}`, cls: "pos" });
}

function onFlag(step) {
  const card = document.getElementById(`card-${step.reportId}`);
  if (card) {
    card.dataset.state = "false";
    document.getElementById(`pill-${step.reportId}`).textContent = "FALSE ALARM";
  }
  addScore(step.reward);
  log({ verb: `flag_false    ${step.reportId.padEnd(8)}`, delta: `+${step.reward.toFixed(2)}`, cls: "pos" });
}

function onAllocate(step) {
  const u = UNITS.find(x => x.id === step.unitId);
  const z = ZONES[step.zone];
  // mark unit deployed
  document.querySelector(`.unit-chip[data-unit="${step.unitId}"]`).dataset.state = "deployed";
  // move dot to zone
  const dot = document.getElementById(`dot-${step.unitId}`);
  dot.style.left = `${z.x}%`;
  dot.style.top  = `${z.y}%`;
  // mark zone active (unless blocked)
  const zoneEl = document.getElementById(`zone-${step.zone}`);
  if (zoneEl.dataset.state !== "blocked") zoneEl.dataset.state = "active";
  addScore(step.reward);
  log({
    verb: `allocate_unit ${u.label} → ${step.zone}`,
    delta: `+${step.reward.toFixed(2)}`,
    cls: step.star ? "pos star" : "pos",
  });
}

function onEvent(step) {
  // banner flash
  const banner = document.createElement("div");
  banner.className = "event-banner";
  banner.textContent = `⚠ ${step.banner}`;
  els.mapPanel.appendChild(banner);
  setTimeout(() => banner.remove(), 1700);

  if (step.kind === "aftershock") {
    const z = document.getElementById(`zone-${step.zone}`);
    z.dataset.state = "blocked";
    document.getElementById(`sev-${step.zone}`).textContent = "BLOCKED";
    log({ verb: `⚠ AFTERSHOCK   ${step.zone.padEnd(10)}`, delta: "blocked", cls: "evt" });
  } else if (step.kind === "road_closure") {
    const [a, b] = step.route;
    const lineA = document.querySelector(`[data-route="${a}__${b}"]`);
    const lineB = document.querySelector(`[data-route="${b}__${a}"]`);
    if (lineA) lineA.dataset.state = "closed";
    if (lineB) lineB.dataset.state = "closed";
    log({ verb: `⚠ ROAD_CLOSURE ${a}↔${b}`, delta: "rerouting", cls: "evt" });
  }
}

function onSitrep(step) {
  addScore(step.reward);
  log({ verb: `publish_sitrep`, delta: `+${step.reward.toFixed(2)}`, cls: "pos star" });
}

function onComplete(step) {
  log({ verb: `══ EPISODE COMPLETE ══`, delta: `score`, cls: "evt" });
  log({ verb: `final_grader_score`, delta: step.score.toFixed(3), cls: "pos star" });
  // ease score to final value
  updateScore(step.score);
}

// ---------------- helpers ----------------

function log({ verb, delta, cls = "" }) {
  const ts = `[${String(state.i + 1).padStart(2, "0")}/${String(STEPS.length).padStart(2, "0")}]`;
  const line = document.createElement("div");
  line.className = `log-line ${cls}`;
  line.innerHTML = `
    <span class="ts">${ts}</span>
    <span class="verb">${verb}</span>
    <span class="delta">${delta}</span>
  `;
  els.logBody.appendChild(line);
  els.logBody.scrollTop = els.logBody.scrollHeight;
}

function addScore(d) { updateScore(state.score + d); }

function updateScore(v) {
  state.score = v;
  els.scoreVal.textContent = v.toFixed(3);
  const pct = Math.max(0, Math.min(100, v * 100));
  els.scoreFill.style.right = `${100 - pct}%`;
}

function bumpReportsCount() {
  const seen = els.reportsList.querySelectorAll(".report-card").length;
  document.getElementById("reports-count").textContent = `${seen} / ${Object.keys(REPORTS).length}`;
}

function updateStepCounter() {
  const total = STEPS.length;
  const cur = Math.min(state.i, total);
  els.stepCounter.innerHTML = `STEP <b>${String(cur).padStart(2, "0")}</b> / ${String(total).padStart(2, "0")}`;
}

function refreshNextAction() {
  const next = STEPS[state.i];
  const p = nextActionPreview(next) || { verb: "—", detail: "—", meta: ["—", "—"] };
  els.naVerb.textContent = p.verb;
  els.naDetail.textContent = p.detail;
  els.naMeta.innerHTML = p.meta.map(m => `<span>${m}</span>`).join("");
}

// ---------------- go ----------------

document.addEventListener("DOMContentLoaded", init);
