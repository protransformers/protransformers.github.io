"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COLORS = { vanilla: "#64748b", pro: "#7c5cff", pro2: "#22d3ee", base: "#475569", other: "#94a3b8", at: "#fbbf24" };

/* ------------------------------------------------------------------ math */
function renderMath() {
  if (!window.katex) return;
  $$(".eq").forEach(el => katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false }));
}

/* ------------------------------------------------------------------ copy buttons */
function initCopy() {
  $$(".copy").forEach(btn => btn.addEventListener("click", async () => {
    const text = btn.dataset.copy || $("#" + btn.dataset.copyTarget).innerText;
    try { await navigator.clipboard.writeText(text); } catch (e) { return; }
    const old = btn.textContent;
    btn.textContent = "Copied ✓"; btn.classList.add("done");
    setTimeout(() => { btn.textContent = old; btn.classList.remove("done"); }, 1400);
  }));
}

/* ------------------------------------------------------------------ ProAttention (same rule as protransformers/pro_attention.py) */
function robustWeights(dists, norm, hp, eps = 1e-2) {
  return dists.map(d => {
    if (norm === "L1") return 1 / (d + eps);
    if (norm === "Huber") return Math.min(hp / (d + eps), 1);
    return Math.max(1 / (d + eps) - 1 / hp, eps); // MCP
  });
}

function proAttention(points, attn, norm, hp, K) {
  const avg = w => {
    const s = w.reduce((a, b) => a + b, 0);
    return [points.reduce((a, p, j) => a + w[j] * p.x, 0) / s, points.reduce((a, p, j) => a + w[j] * p.y, 0) / s];
  };
  let z = avg(attn);
  const path = [z];
  let w = attn.slice();
  for (let k = 0; k < K; k++) {
    const d = points.map(p => Math.hypot(p.x - z[0], p.y - z[1]));
    w = robustWeights(d, norm, hp).map((v, j) => v * attn[j]);
    z = avg(w);
    path.push(z);
  }
  const s = w.reduce((a, b) => a + b, 0);
  return { z, path, w: w.map(v => v / s) };
}

/* ------------------------------------------------------------------ interactive playground */
function initPlayground() {
  const canvas = $("#irls"), ctx = canvas.getContext("2d");
  const state = { norm: "MCP", K: 3, gamma: 4, delta: 1.5, points: [], drag: null };
  const W = { x0: -5.5, x1: 5.5, y0: -3.4, y1: 3.4 };
  let rng = 7;
  const rand = () => (rng = (rng * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());

  function reset() {
    rng = 7;
    state.points = Array.from({ length: 14 }, () => ({ x: -2 + 0.9 * gauss(), y: -0.3 + 0.9 * gauss(), adv: false }));
    draw();
  }
  function attack() {
    const cx = 2 + rand() * 2.5, cy = (rand() - 0.5) * 4.5;
    for (let i = 0; i < 3; i++) state.points.push({ x: cx + 0.35 * gauss(), y: cy + 0.35 * gauss(), adv: true });
    draw();
  }

  let size = { w: 0, h: 0, s: 1 };
  function resize() {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    size = { w: r.width, h: r.height, s: Math.min(r.width / (W.x1 - W.x0), r.height / (W.y1 - W.y0)) };
    draw();
  }
  const toPx = (x, y) => [size.w / 2 + x * size.s, size.h / 2 - y * size.s];
  const toW = (px, py) => [(px - size.w / 2) / size.s, -(py - size.h / 2) / size.s];

  function draw() {
    if (!size.w) return;
    const { points } = state;
    ctx.clearRect(0, 0, size.w, size.h);
    // grid
    ctx.strokeStyle = "rgba(38,44,64,.55)"; ctx.lineWidth = 1;
    for (let gx = Math.ceil(W.x0 * 2) / 2; gx <= W.x1 * 2; gx += 1) {
      const [px] = toPx(gx, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, size.h); ctx.stroke();
    }
    for (let gy = -8; gy <= 8; gy += 1) {
      const [, py] = toPx(0, gy); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(size.w, py); ctx.stroke();
    }
    if (!points.length) return;

    const attn = points.map(() => 1);
    const hp = state.norm === "Huber" ? state.delta : state.gamma;
    const vanilla = proAttention(points, attn, "L2", hp, 0).z;
    const pro = proAttention(points, attn, state.norm, hp, state.K);
    const clean = points.filter(p => !p.adv);
    const ref = clean.length ? proAttention(clean, clean.map(() => 1), "L2", hp, 0).z : vanilla;

    // MCP / Huber radius around the robust estimate
    if (state.norm !== "L1") {
      const [cx, cy] = toPx(...pro.z);
      ctx.beginPath(); ctx.arc(cx, cy, hp * size.s, 0, 2 * Math.PI);
      ctx.setLineDash([5, 6]); ctx.strokeStyle = "rgba(251,191,36,.35)"; ctx.stroke(); ctx.setLineDash([]);
    }
    // lines from tokens to the robust estimate, opacity = weight
    const wmax = Math.max(...pro.w);
    points.forEach((p, j) => {
      const a = toPx(p.x, p.y), b = toPx(...pro.z);
      ctx.strokeStyle = `rgba(251,191,36,${0.05 + 0.35 * pro.w[j] / wmax})`;
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    });
    // tokens, radius = robust weight
    points.forEach((p, j) => {
      const [x, y] = toPx(p.x, p.y);
      const r = 3 + 9 * Math.sqrt(pro.w[j] / wmax);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = p.adv ? "rgba(248,113,113,.85)" : "rgba(96,165,250,.85)"; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = p.adv ? "#fecaca" : "#bfdbfe"; ctx.stroke();
    });
    // IRLS path
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
    ctx.beginPath(); pro.path.forEach((z, i) => { const [x, y] = toPx(...z); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    ctx.setLineDash([]);
    pro.path.slice(1, -1).forEach(z => { const [x, y] = toPx(...z); ctx.fillStyle = "rgba(251,191,36,.6)"; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); });
    // vanilla estimate (✕)
    const [vx, vy] = toPx(...vanilla);
    ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.moveTo(vx - 8, vy - 8); ctx.lineTo(vx + 8, vy + 8); ctx.moveTo(vx + 8, vy - 8); ctx.lineTo(vx - 8, vy + 8); ctx.stroke();
    // robust estimate (◆)
    const [rx, ry] = toPx(...pro.z);
    ctx.fillStyle = "#fbbf24"; ctx.strokeStyle = "#0b0d14"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(rx, ry - 11); ctx.lineTo(rx + 11, ry); ctx.lineTo(rx, ry + 11); ctx.lineTo(rx - 11, ry); ctx.closePath(); ctx.fill(); ctx.stroke();

    if (!points.some(p => p.adv)) {
      ctx.fillStyle = "rgba(154,163,186,.9)"; ctx.font = "600 14px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Click anywhere to inject an adversarial token →", size.w / 2, 30);
    }
    $("#shift-l2").textContent = Math.hypot(vanilla[0] - ref[0], vanilla[1] - ref[1]).toFixed(2);
    $("#shift-pro").textContent = Math.hypot(pro.z[0] - ref[0], pro.z[1] - ref[1]).toFixed(2);
  }

  function pos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  canvas.addEventListener("pointerdown", e => {
    const [px, py] = pos(e);
    const hit = state.points.findIndex(p => { const [x, y] = toPx(p.x, p.y); return Math.hypot(x - px, y - py) < 14; });
    if (hit >= 0) state.drag = hit;
    else { const [x, y] = toW(px, py); state.points.push({ x, y, adv: true }); state.drag = state.points.length - 1; }
    canvas.setPointerCapture(e.pointerId);
    draw();
  });
  canvas.addEventListener("pointermove", e => {
    if (state.drag === null) return;
    const [x, y] = toW(...pos(e));
    Object.assign(state.points[state.drag], { x, y });
    draw();
  });
  const end = () => { state.drag = null; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);

  const hpLabel = () => {
    const huber = state.norm === "Huber";
    $("#hp-label").style.display = $("#gamma").style.display = state.norm === "L1" ? "none" : "";
    $("#hp-label").innerHTML = huber ? `δ (Huber) = <span id="gval">${state.delta.toFixed(1)}</span>` : `γ (MCP) = <span id="gval">${state.gamma.toFixed(1)}</span>`;
    $("#gamma").value = huber ? state.delta : state.gamma;
  };
  $$("#norm button").forEach(b => b.addEventListener("click", () => {
    $$("#norm button").forEach(x => x.classList.toggle("on", x === b));
    state.norm = b.dataset.v; hpLabel(); draw();
  }));
  $("#K").addEventListener("input", e => { state.K = +e.target.value; $("#kval").textContent = state.K; draw(); });
  $("#gamma").addEventListener("input", e => {
    if (state.norm === "Huber") state.delta = +e.target.value; else state.gamma = +e.target.value;
    $("#gval").textContent = (+e.target.value).toFixed(1); draw();
  });
  $("#add-adv").addEventListener("click", attack);
  $("#reset").addEventListener("click", reset);
  new ResizeObserver(resize).observe(canvas);
  reset();
  attack();
}

/* ------------------------------------------------------------------ tiny SVG charts */
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, text) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
function legend(container, series) {
  const d = document.createElement("div");
  d.className = "chart-legend";
  d.innerHTML = series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  container.appendChild(d);
}

function barChart(container, { groups, series, yMax = 100, unit = "%", height = 240 }) {
  container.innerHTML = "";
  const W = 560, H = height, L = 34, B = 26, T = 18;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  for (let v = 0; v <= yMax; v += yMax / 4) {
    const y = H - B - (H - B - T) * v / yMax;
    svg.append(el("line", { x1: L, x2: W, y1: y, y2: y, class: "grid" }), el("text", { x: L - 6, y: y + 4, "text-anchor": "end" }, v));
  }
  const gw = (W - L) / groups.length, bw = Math.min(30, (gw - 16) / series.length);
  groups.forEach((g, i) => {
    const x0 = L + i * gw + (gw - bw * series.length) / 2;
    series.forEach((s, j) => {
      const v = s.values[i], h = (H - B - T) * v / yMax, x = x0 + j * bw;
      const r = el("rect", { x: x + 2, width: bw - 4, y: H - B, height: 0, rx: 3, fill: s.color });
      r.append(el("title", {}, `${s.name}: ${v}${unit}`));
      svg.append(r);
      requestAnimationFrame(() => { r.style.transition = "all .6s cubic-bezier(.2,.8,.2,1)"; r.setAttribute("y", H - B - h); r.setAttribute("height", h); });
      if (series.length <= 3 || s.label) svg.append(el("text", { x: x + bw / 2, y: H - B - h - 5, "text-anchor": "middle", class: "val" }, v));
    });
    svg.append(el("text", { x: L + i * gw + gw / 2, y: H - 6, "text-anchor": "middle" }, g));
  });
  container.appendChild(svg);
  legend(container, series);
}

function lineChart(container, { xs, series, yMin = 0, yMax = 100, xLabel = "" }) {
  container.innerHTML = "";
  const W = 560, H = 250, L = 34, B = 34, T = 14, R = 10;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  const X = i => L + (W - L - R) * i / (xs.length - 1);
  const Y = v => H - B - (H - B - T) * (v - yMin) / (yMax - yMin);
  for (let v = yMin; v <= yMax; v += (yMax - yMin) / 4) svg.append(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" }), el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v));
  const step = Math.ceil(xs.length / 8);
  xs.forEach((x, i) => { if (i % step === 0 || i === xs.length - 1) svg.append(el("text", { x: X(i), y: H - B + 16, "text-anchor": "middle" }, x)); });
  if (xLabel) svg.append(el("text", { x: (L + W) / 2, y: H - 2, "text-anchor": "middle" }, xLabel));
  series.forEach(s => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("");
    const p = el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.bold ? 3 : 2, "stroke-dasharray": s.dash || "" });
    svg.append(p);
    s.values.forEach((v, i) => { const c = el("circle", { cx: X(i), cy: Y(v), r: s.bold ? 3.5 : 2.5, fill: s.color }); c.append(el("title", {}, `${s.name} @ ${xs[i]}: ${v}`)); svg.append(c); });
  });
  container.appendChild(svg);
  legend(container, series);
}

/* ------------------------------------------------------------------ results from the paper */
function initCharts() {
  barChart($("#chart-text"), {
    groups: ["TextFooler", "TextBugger", "DeepWordBug", "PWWS"],
    series: [
      { name: "BERT", color: COLORS.vanilla, values: [19.7, 31.7, 37.5, 43.1] },
      { name: "BERT + FreeLB (adv. training)", color: COLORS.other, values: [38.0, 42.8, 56.1, 57.0] },
      { name: "Pro-BERT (MCP), no training", color: COLORS.pro, values: [39.2, 48.3, 51.8, 56.2] },
      { name: "Pro-BERT (MCP) + AT", color: COLORS.pro2, values: [56.8, 60.7, 61.0, 68.8] },
    ],
  });
  barChart($("#chart-jb"), {
    groups: ["Vicuna", "Pro-Vicuna δ=0.1", "δ=0.2", "δ=0.3", "δ=0.5"],
    series: [{ name: "Attack success rate (%), lower is better", color: COLORS.pro, values: [91.8, 1.8, 8.2, 21.8, 36.4] }],
    height: 210,
  });
  // highlight the vanilla bar
  const jb = $("#chart-jb rect"); if (jb) jb.setAttribute("fill", "#f87171");
  lineChart($("#chart-t5"), {
    xs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], xLabel: "# perturbed words", yMin: 20, yMax: 100,
    series: [
      { name: "T5", color: COLORS.vanilla, values: [89.1, 89.1, 89.1, 73.3, 69.3, 63.4, 47.5, 47.5, 43.6, 36.6, 31.7, 29.7, 29.7] },
      { name: "Pro-T5 (ℓ₁)", color: COLORS.other, dash: "4 4", values: [90.1, 90.1, 90.1, 67.3, 64.4, 54.5, 54.5, 51.5, 46.5, 46.5, 40.6, 39.6, 39.6] },
      { name: "Pro-T5 (MCP)", color: COLORS.pro, bold: true, values: [87.1, 87.1, 87.1, 71.3, 69.3, 64.4, 64.4, 64.4, 64.4, 63.3, 63.3, 63.3, 61.4] },
    ],
  });
  lineChart($("#chart-graph"), {
    xs: ["0%", "10%", "20%", "30%", "40%"], xLabel: "perturbed edges", yMin: 40, yMax: 90,
    series: [
      { name: "GCN", color: COLORS.base, dash: "4 4", values: [85.0, 69.6, 60.9, 54.2, 48.4] },
      { name: "SoftMedian", color: COLORS.other, dash: "4 4", values: [85.0, 75.5, 69.5, 62.8, 58.1] },
      { name: "GAT", color: COLORS.vanilla, values: [83.5, 71.2, 65.0, 60.5, 56.7] },
      { name: "Pro-GAT", color: COLORS.pro, bold: true, values: [84.6, 75.5, 72.1, 69.0, 66.5] },
    ],
  });

  const VIS = {
    budgets: ["clean", "1/255", "4/255", "8/255"],
    models: { DeiT: [97.91, 38.98, 0.44, 0.0], ConViT: [98.70, 41.75, 1.83, 0.0], BEiT: [97.87, 6.81, 0.0, 0.0], Swin: [98.30, 14.89, 0.02, 0.01], ViT: [98.74, 34.61, 1.83, 0.26], "Pro-ViT": [98.40, 77.39, 48.11, 33.40] },
  };
  const visDraw = () => {
    const i = +$("#vis-eps").value, names = Object.keys(VIS.models);
    $("#vis-eps-label").textContent = `ε = ${VIS.budgets[i]}`;
    barChart($("#chart-vision"), {
      groups: names,
      series: [{ name: `Robust accuracy (%) at ${VIS.budgets[i]}`, color: COLORS.vanilla, values: names.map(n => VIS.models[n][i]) }],
    });
    const bars = $$("#chart-vision rect");
    bars[bars.length - 1].setAttribute("fill", COLORS.pro);
  };
  $("#vis-eps").addEventListener("input", visDraw);
  visDraw();
}

/* ------------------------------------------------------------------ tabs */
function initTabs() {
  $$(".tab").forEach(t => t.addEventListener("click", () => {
    $$(".tab").forEach(x => x.classList.toggle("on", x === t));
    $$(".panel").forEach(p => p.classList.toggle("on", p.id === "tab-" + t.dataset.tab));
    if (t.dataset.tab === "vision") $("#vis-eps").dispatchEvent(new Event("input"));
  }));
}

/* ------------------------------------------------------------------ language: real TextFooler examples */
const AG = ["World", "Sports", "Business", "Sci/Tech"];
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function initText() {
  let data;
  try { data = await (await fetch("data/text_examples.json")).json(); } catch (e) { return; }
  const argmax = p => p.indexOf(Math.max(...p));
  const ok = x => AG[argmax(x.pro_adv)] === x.label;
  // Show the short, readable examples first
  data = data.map((x, i) => ({ ...x, i, n: x.orig.split(/\s+/).length })).sort((a, b) => (a.n > 60) - (b.n > 60) || ok(b) - ok(a) || a.i - b.i);
  const proOk = data.filter(ok).length;
  $("#tx-caption").textContent = `All ${data.length} sentences TextFooler crafted to fool vanilla BERT. Pro-BERT (MCP), with the same weights, still gets ${proOk}/${data.length} right.`;
  let idx = 0, view = "adv";

  const verdict = (name, probs, label) => {
    const pred = argmax(probs), ok = AG[pred] === label;
    return [`<div class="who">${name}<span class="tag ${ok ? "tag-good" : "tag-bad"}">${ok ? "correct" : "fooled"}</span></div>` +
      AG.map((c, k) => `<div class="bar ${c === label ? "true" : ""} ${k === pred ? "pred" : ""}"><span>${c}</span><div class="track"><div class="fill" style="width:${(100 * probs[k]).toFixed(1)}%"></div></div><span>${(100 * probs[k]).toFixed(0)}%</span></div>`).join(""), ok];
  };

  function show() {
    const x = data[idx];
    $("#tx-idx").textContent = `${idx + 1} / ${data.length}`;
    // highlight substituted words
    const swaps = new Map(x.swaps.map(([a, b]) => [view === "adv" ? b : a, view === "adv" ? a : b]));
    const text = view === "adv" ? x.adv : x.orig;
    $("#tx-sentence").innerHTML = `<span class="muted small">Label: <b>${x.label}</b></span><br>` + esc(text).replace(/[A-Za-z][\w'-]*/g, w => {
      if (!swaps.has(w)) return w;
      return `<span class="${view === "adv" ? "swap" : "was"}" title="${view === "adv" ? "was" : "becomes"}: ${esc(swaps.get(w))}">${w}</span>`;
    });
    const [hb, okb] = verdict("BERT", view === "adv" ? x.bert_adv : x.bert_orig, x.label);
    const [hp, okp] = verdict("Pro-BERT (MCP)", view === "adv" ? x.pro_adv : x.pro_orig, x.label);
    $("#tx-bert").innerHTML = hb; $("#tx-bert").className = "verdict " + (okb ? "ok" : "fail");
    $("#tx-pro").innerHTML = hp; $("#tx-pro").className = "verdict " + (okp ? "ok" : "fail");
  }
  $("#tx-prev").addEventListener("click", () => { idx = (idx - 1 + data.length) % data.length; show(); });
  $("#tx-next").addEventListener("click", () => { idx = (idx + 1) % data.length; show(); });
  $$("#tx-view button").forEach(b => b.addEventListener("click", () => {
    $$("#tx-view button").forEach(x => x.classList.toggle("on", x === b)); view = b.dataset.v; show();
  }));
  show();
}

/* ------------------------------------------------------------------ vision: patch toy */
function initPatches() {
  const grid = $("#patch-grid"), adv = new Set([10, 11, 19, 37, 45, 52]);
  for (let i = 0; i < 64; i++) { const d = document.createElement("div"); if (adv.has(i)) d.className = "adv"; grid.appendChild(d); }
  grid.parentElement.addEventListener("mouseenter", () => grid.classList.add("robust"));
  grid.parentElement.addEventListener("mouseleave", () => grid.classList.remove("robust"));
  grid.addEventListener("click", () => grid.classList.toggle("robust"));
}

/* ------------------------------------------------------------------ graph: toy adversarial edges */
function initGraph() {
  const svg = $("#graph-svg");
  const T = { x: 200, y: 160 };
  // 1-D node features: class A ≈ -1, class B ≈ +1; the attacker links T to confidently-B nodes (≈ +2.5)
  const clean = [[95, 70, -1.1], [70, 160, -0.8], [95, 250, -1.2], [170, 40, -0.9], [180, 285, -1.0]].map(([x, y, f]) => ({ x, y, f }));
  const advN = [[360, 60], [430, 45], [480, 105], [400, 135], [470, 175], [380, 205], [440, 245], [490, 255], [370, 270]].map(([x, y], i) => ({ x, y, f: 2.4 + 0.1 * ((i * 7) % 3) }));
  const selfF = -0.7;

  function predict(k, robust) {
    const pts = [{ x: selfF, y: 0 }, ...clean.map(n => ({ x: n.f, y: 0 })), ...advN.slice(0, k).map(n => ({ x: n.f, y: 0 }))];
    const z = proAttention(pts, pts.map(() => 1), robust ? "MCP" : "L2", 1.5, robust ? 3 : 0).z[0];
    return z;
  }
  function draw() {
    const k = +$("#g-edges").value;
    $("#g-k").textContent = k;
    svg.innerHTML = "";
    const line = (a, cls, adv) => svg.append(el("line", { x1: T.x, y1: T.y, x2: a.x, y2: a.y, stroke: adv ? "#f87171" : "#3b4a6b", "stroke-width": adv ? 2.5 : 2, "stroke-dasharray": adv ? "6 5" : "" }));
    clean.forEach(n => line(n));
    advN.slice(0, k).forEach(n => line(n, "", true));
    const node = (n, fill, label, r = 16) => { svg.append(el("circle", { cx: n.x, cy: n.y, r, fill, stroke: "#0b0d14", "stroke-width": 3 })); if (label) svg.append(el("text", { x: n.x, y: n.y + 4, "text-anchor": "middle" }, label)); };
    clean.forEach(n => node(n, "#60a5fa", "A"));
    advN.forEach((n, i) => node(n, i < k ? "#f87171" : "#3a2530", "B"));
    const zG = predict(k, false), zP = predict(k, true), mid = 0;
    node(T, "#e2e8f0", "T", 21);
    svg.append(el("text", { x: 440, y: 305, "text-anchor": "middle", style: "fill:#9aa3ba;font-weight:500" }, "class-B nodes (attacker)"));
    svg.append(el("text", { x: 120, y: 305, "text-anchor": "middle", style: "fill:#9aa3ba;font-weight:500" }, "true neighbours (class A)"));
    const box = (id, name, z) => {
      const ok = z < mid;
      $(id).className = "verdict " + (ok ? "ok" : "fail");
      $(id).innerHTML = `<div class="who">${name}<span class="tag ${ok ? "tag-good" : "tag-bad"}">${ok ? "predicts A ✓" : "predicts B ✗"}</span></div>` +
        `<div class="muted small">aggregated feature ${z.toFixed(2)} (class A ≈ −1, class B ≈ +1)</div>`;
    };
    box("#g-gat", "GAT", zG);
    box("#g-pro", "Pro-GAT (MCP)", zP);
  }
  $("#g-edges").addEventListener("input", draw);
  draw();
}

/* ------------------------------------------------------------------ boot */
document.addEventListener("DOMContentLoaded", () => {
  renderMath();
  initCopy();
  initTabs();
  initPlayground();
  initCharts();
  initText();
  initPatches();
  initGraph();
});
