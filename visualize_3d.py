"""
3D market explorer for the TDA crash project.

Each line = cumulative return of an index or sector ETF from the start of the selected range.
  green = gain, red = loss (brighter = bigger move)
  cyan line (front)   = TDA signal (L1 norm of the persistence landscape), scaled to fit
  gold curtains       = crash dates
  glass plane         = 0% return

Lines are sorted front to back from worst to best performer in the selected range.

Layout: price chart of the selected ticker on top (drag to scroll through time, wheel to zoom),
3D view underneath that follows whatever range the top chart shows.

Controls in the page:
  - ticker dropdown (or click a name in the side panel)
  - left/right arrow keys or the arrow buttons step through time
  - search box: "2008", "2008-09", "Mar 2020", "Nov 5 2026", "2007-06 to 2009-03", "2020-02-01..2020-06-30"
  - From / To pickers, quick ranges (1M ... All), one-click crash buttons
  - toggles for sectors, TDA line, zero plane

Run from the folder with tda_crashes.py:
  pip install plotly
  python visualize_3d.py
  python visualize_3d.py --start 2007-06-01 --end 2009-06-01
  python visualize_3d.py --no-open
"""

import argparse
import json
import os
import webbrowser

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from tda_crashes import load_prices, log_returns, compute_norms, CRASHES, OUT_DIR, WINDOW

SECTORS = {
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Cons. Disc.", "XLP": "Cons. Staples", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Comm. Services",
}
SECTOR_FILE = "sectors.csv"
OUT_FILE = "market_explorer.html"


# ---------------- data ----------------
def load_sectors(start):
    """Sector ETF closes, cached. Returns an empty frame if the download fails."""
    if os.path.exists(SECTOR_FILE):
        return pd.read_csv(SECTOR_FILE, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
        raw = yf.download(list(SECTORS), start=start, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("empty download")
        df = raw["Close"][[t for t in SECTORS if t in raw["Close"].columns]]
        df.columns = [f"{SECTORS[t]} ({t})" for t in df.columns]
        df.to_csv(SECTOR_FILE)
        return df
    except Exception as e:
        print(f"Sector download failed ({e}); showing the 4 indices only.")
        return pd.DataFrame()


def load_norms(prices):
    path = os.path.join(OUT_DIR, "landscape_norms.csv")
    if os.path.exists(path):
        norms = pd.read_csv(path, index_col=0, parse_dates=True)
        if norms.index[-1] >= prices.index[-1]:
            return norms
    print("Computing landscape norms (run tda_crashes.py first to skip this)...")
    norms = compute_norms(log_returns(prices))
    norms.to_csv(path)
    return norms


def clean(values):
    return [None if not np.isfinite(v) else round(float(v), 4) for v in values]


def build_payload(prices, sectors, norms, start, end):
    idx = norms.index
    series = []
    for col in prices.columns:
        series.append({"name": col, "kind": "index", "p": clean(prices[col].reindex(idx).ffill().values)})
    for col in sectors.columns:
        s = sectors[col].reindex(idx)
        first = s.first_valid_index()
        s = s.ffill()
        if first is not None:
            s[s.index < first] = np.nan          # don't forward-fill before the ETF existed
        series.append({"name": col, "kind": "sector", "p": clean(s.values)})
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "series": series,
        "l1": [float(f"{v:.6g}") for v in norms["L1"].values],
        "crashes": [{"name": k, "date": v} for k, v in CRASHES.items()],
        "init": {"start": start, "end": end},
        "window": WINDOW,
    }


# ---------------- page ----------------
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Crash Explorer</title>
<script>__PLOTLY__</script>
<style>
:root{--bg:#0b0e14;--panel:#121722;--line:#1e2531;--text:#e8ecf1;--muted:#8a94a3;--green:#19e57a;--red:#ff3b4e;--cyan:#29d3ff;--gold:#ffc83d}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line)}
.title{font-size:15px;font-weight:650;margin-right:6px}
.group{display:flex;gap:5px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:4px 6px}
.group label{color:var(--muted);font-size:12px}
input[type=text],input[type=date],select{background:#080b10;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 7px;font:inherit;color-scheme:dark}
input[type=text]{width:230px}
select{font-weight:600}
button{background:#171d29;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font:inherit;cursor:pointer;white-space:nowrap}
button:hover{border-color:#36445a}
button.primary{background:#1d6fff;border-color:#1d6fff;color:#fff}
button.on{border-color:var(--cyan);color:var(--cyan)}
.chip{font-size:12px;padding:3px 9px;border-radius:999px}
.chip .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--gold);margin-right:6px;vertical-align:1px}
.err{color:var(--red);font-size:12px}
main{flex:1;display:grid;grid-template-columns:1fr 250px;min-height:0}
.charts{display:flex;flex-direction:column;min-height:0}
.pane-h{display:flex;justify-content:space-between;align-items:center;padding:6px 12px 0;color:var(--muted);font-size:12px}
.pane-h b{color:var(--text);font-weight:600}
#top{height:32vh;min-height:220px;border-bottom:1px solid var(--line)}
#chart{flex:1;min-height:380px}
aside{border-left:1px solid var(--line);padding:12px 14px;overflow:auto}
aside h3{margin:14px 0 6px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
aside h3:first-child{margin-top:0}
.stat{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px dashed var(--line)}
.stat span:first-child{color:var(--muted)}
.rank{display:flex;justify-content:space-between;padding:2px 4px;border-radius:4px;cursor:pointer;font-variant-numeric:tabular-nums}
.rank:hover{background:#161c27}
.rank.sel{background:#1a2333}
.key{color:var(--muted);line-height:1.7;margin-top:12px}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:8px;vertical-align:3px}
.hint{color:var(--muted);font-size:11px;margin-top:12px;line-height:1.5}
kbd{background:#171d29;border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:11px}
@media(max-width:950px){main{grid-template-columns:1fr}aside{border-left:none;border-top:1px solid var(--line)}}
</style></head><body>

<div class="bar">
  <span class="title">Market Crash Explorer</span>
  <div class="group"><select id="ticker"></select></div>
  <div class="group">
    <input type="text" id="q" placeholder="Search: 2008 · Mar 2020 · 2007-06 to 2009-03">
    <button class="primary" id="go">Go</button><span class="err" id="err"></span>
  </div>
  <div class="group">
    <label>From</label><input type="date" id="from"><label>To</label><input type="date" id="to">
    <button id="apply">Apply</button>
  </div>
  <div class="group">
    <button id="back" title="Scroll back (←)">◀</button>
    <span id="quick" style="display:flex;gap:5px"></span>
    <button id="fwd" title="Scroll forward (→)">▶</button>
  </div>
  <div class="group">
    <button id="tSec" class="on">Sectors</button>
    <button id="tTda" class="on">TDA</button>
    <button id="tPlane" class="on">Plane</button>
    <button id="resetCam">Reset 3D</button>
  </div>
</div>
<div class="bar" id="crashBar"></div>

<main>
  <div class="charts">
    <div class="pane-h"><span><b id="tName"></b> · daily close</span><span id="tChg"></span></div>
    <div id="top"></div>
    <div class="pane-h"><span><b>3D returns</b> · cumulative return from range start, worst in front → best in back</span><span id="sub"></span></div>
    <div id="chart"></div>
  </div>
  <aside>
    <h3>Selected range</h3>
    <div class="stat"><span>From</span><span id="sFrom"></span></div>
    <div class="stat"><span>To</span><span id="sTo"></span></div>
    <div class="stat"><span>Trading days</span><span id="sDays"></span></div>
    <div class="stat"><span id="sTk"></span><span id="sChg"></span></div>
    <div class="stat"><span>Max drawdown</span><span id="sDD"></span></div>
    <div class="stat"><span>TDA peak</span><span id="sPeak"></span></div>
    <h3>Performance</h3>
    <div id="ranks"></div>
    <div class="key">
      <div><span class="sw" style="background:var(--green)"></span>gain from range start</div>
      <div><span class="sw" style="background:var(--red)"></span>loss from range start</div>
      <div><span class="sw" style="background:#fff"></span>selected ticker (thick)</div>
      <div><span class="sw" style="background:var(--cyan)"></span>TDA signal (L1, scaled)</div>
      <div><span class="sw" style="background:var(--gold)"></span>crash date</div>
    </div>
    <div class="hint">Top chart: drag to scroll through time, scroll wheel to zoom, double-click to reset. The 3D view follows. <kbd>←</kbd> <kbd>→</kbd> step through time. Click a name above to switch the ticker.</div>
  </aside>
</main>

<script>
const D = __DATA__;
const N = D.dates.length, MAX_PTS = 1500;
const $ = id => document.getElementById(id);
const CAM = {eye: {x: -1.5, y: -1.8, z: 0.6}, center: {x: 0.05, y: 0.05, z: -0.1}};
let range = [0, N - 1], show = {sec: true, tda: true, plane: true}, tk = 0, syncing = false, timer = null;

// ---------------- dates + search ----------------
const toDate = s => new Date(s + "T00:00:00Z"), iso = d => d.toISOString().slice(0, 10), pad = n => String(n).padStart(2, "0");
function idxOf(s, side){
  let lo = 0, hi = N;
  while (lo < hi){ const m = (lo + hi) >> 1; if (D.dates[m] < s) lo = m + 1; else hi = m; }
  if (side === "hi"){ if (lo < N && D.dates[lo] === s) return lo; return Math.max(0, lo - 1); }
  return Math.min(N - 1, lo);
}
const MONTHS = {jan:1,feb:2,mar:3,apr:4,may:5,jun:6,jul:7,aug:8,sep:9,oct:10,nov:11,dec:12};
const lastDay = (y, m) => new Date(Date.UTC(y, m, 0)).getUTCDate();
function parsePart(p){
  p = p.trim().toLowerCase().replace(/,/g, " ").replace(/\s+/g, " "); let m;
  if ((m = p.match(/^(\d{4})$/))) return [`${m[1]}-01-01`, `${m[1]}-12-31`];
  if ((m = p.match(/^(\d{4})[-/](\d{1,2})$/))){ const y=+m[1], mo=+m[2]; return [`${y}-${pad(mo)}-01`, `${y}-${pad(mo)}-${pad(lastDay(y,mo))}`]; }
  if ((m = p.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$/))){ const s=`${m[1]}-${pad(m[2])}-${pad(m[3])}`; return [s, s]; }
  if ((m = p.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/))){ const s=`${m[3]}-${pad(m[1])}-${pad(m[2])}`; return [s, s]; }
  if ((m = p.match(/^([a-z]{3,9})\.? (\d{4})$/)) && MONTHS[m[1].slice(0,3)]){ const mo=MONTHS[m[1].slice(0,3)], y=+m[2]; return [`${y}-${pad(mo)}-01`, `${y}-${pad(mo)}-${pad(lastDay(y,mo))}`]; }
  if ((m = p.match(/^([a-z]{3,9})\.? (\d{1,2}) (\d{4})$/)) && MONTHS[m[1].slice(0,3)]){ const s=`${m[3]}-${pad(MONTHS[m[1].slice(0,3)])}-${pad(m[2])}`; return [s, s]; }
  return null;
}
function parseQuery(q){
  const parts = q.split(/\s*(?:\bto\b|\.\.|–|—|\bthrough\b|\buntil\b)\s*/i).filter(Boolean);
  if (parts.length === 1){
    const a = parsePart(parts[0]); if (!a) return null; if (a[0] !== a[1]) return a;
    const s = toDate(a[0]), e = toDate(a[0]); s.setUTCMonth(s.getUTCMonth() - 1); e.setUTCMonth(e.getUTCMonth() + 1);
    return [iso(s), iso(e)];
  }
  if (parts.length === 2){ const a = parsePart(parts[0]), b = parsePart(parts[1]); return a && b ? [a[0], b[1]] : null; }
  return null;
}
function setRange(s, e){
  let i = idxOf(s, "lo"), j = idxOf(e, "hi");
  if (j <= i){ $("err").textContent = "No trading days in that range"; return; }
  if (j - i < 5){ i = Math.max(0, i - 3); j = Math.min(N - 1, j + 3); }
  $("err").textContent = ""; apply(i, j, "control");
}
function shift(dir){
  const w = range[1] - range[0], step = Math.max(1, Math.round(w * 0.25));
  let i = range[0] + dir * step, j = range[1] + dir * step;
  if (i < 0){ j -= i; i = 0; } if (j > N - 1){ i -= j - (N - 1); j = N - 1; }
  apply(Math.max(0, i), j, "control");
}

// ---------------- colors ----------------
const RG = [[0,"#ff1e3c"],[0.25,"#ff5468"],[0.46,"#ffb3bb"],[0.5,"#eef2f5"],[0.54,"#b4f7cf"],[0.75,"#3cf08f"],[1,"#00d26a"]];
function rgAt(v, lim){
  const t = Math.max(0, Math.min(1, 0.5 + 0.5 * v / lim));
  for (let k = 1; k < RG.length; k++) if (t <= RG[k][0]){
    const [t0, c0] = RG[k-1], [t1, c1] = RG[k], f = (t - t0) / (t1 - t0 || 1);
    const a = [1,3,5].map(o => parseInt(c0.substr(o, 2), 16)), b = [1,3,5].map(o => parseInt(c1.substr(o, 2), 16));
    return `rgb(${a.map((x, n) => Math.round(x + (b[n] - x) * f)).join(",")})`;
  }
  return RG[RG.length - 1][1];
}
const fmtPct = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const short = n => n.replace(/ \(.*\)/, "");
function monthLabel(s){ return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][+s.slice(5,7)-1] + " " + s.slice(2,4); }

// ---------------- range changes ----------------
function apply(i, j, src){
  range = [i, j];
  if (src !== "top") syncTop(true); else syncTop(false);
  clearTimeout(timer);
  timer = setTimeout(render3D, src === "top" ? 90 : 0);
  $("from").value = D.dates[i]; $("to").value = D.dates[j];
  $("sub").textContent = `${D.dates[i]} → ${D.dates[j]}`;
  history.replaceState(null, "", `#${D.dates[i]}..${D.dates[j]}&${encodeURIComponent(D.series[tk].name)}`);
}

// ---------------- top price chart ----------------
function initTop(){
  const s = D.series[tk];
  $("tName").textContent = s.name;
  const traces = [
    {x: D.dates, y: s.p, type: "scatter", mode: "lines", name: s.name, line: {color: "#e9edf2", width: 1.3},
     fill: "tozeroy", fillcolor: "rgba(233,237,242,0.04)", hovertemplate: "%{x}<br>%{y:,.2f}<extra></extra>"},
    {x: D.dates, y: D.l1, type: "scatter", mode: "lines", name: "TDA signal", yaxis: "y2", fill: "tozeroy",
     line: {color: "#29d3ff", width: 1}, fillcolor: "rgba(41,211,255,0.13)", hovertemplate: "TDA %{y:.2e}<extra></extra>"},
  ];
  const shapes = D.crashes.map(c => ({type: "line", xref: "x", yref: "paper", x0: c.date, x1: c.date, y0: 0, y1: 1,
                                      line: {color: "#ffc83d", width: 1, dash: "dot"}}));
  const layout = {
    paper_bgcolor: "#0b0e14", plot_bgcolor: "#0b0e14", margin: {l: 12, r: 64, t: 6, b: 26},
    dragmode: "pan", hovermode: "x unified", showlegend: false, font: {color: "#8a94a3", size: 11},
    xaxis: {type: "date", gridcolor: "#161c26", range: [D.dates[range[0]], D.dates[range[1]]], showspikes: true,
            spikecolor: "#55606f", spikethickness: 1, spikedash: "dot", spikemode: "across"},
    yaxis: {side: "right", gridcolor: "#161c26", tickformat: ",.0f", nticks: 5},
    yaxis2: {overlaying: "y", domain: [0, 1], range: [0, 1], showgrid: false, showticklabels: false, fixedrange: true},
    shapes,
  };
  Plotly.react("top", traces, layout, {displaylogo: false, responsive: true, scrollZoom: true,
                                       modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"]});
}
function yRange(arr, i, j){
  let lo = Infinity, hi = -Infinity;
  for (let k = i; k <= j; k++){ const v = arr[k]; if (v == null) continue; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const p = (hi - lo) * 0.08 || 1; return [lo - p, hi + p];
}
function syncTop(setX){
  const [i, j] = range, s = D.series[tk];
  let lmax = 0; for (let k = i; k <= j; k++) lmax = Math.max(lmax, D.l1[k]);
  const upd = {"yaxis.range": yRange(s.p, i, j), "yaxis2.range": [0, (lmax || 1) * 4]};   // TDA sits in the bottom third
  if (setX) upd["xaxis.range"] = [D.dates[i], D.dates[j]];
  syncing = true; Plotly.relayout("top", upd).then(() => { syncing = false; });
  const a = s.p.slice(i, j + 1).find(v => v != null), b = [...s.p.slice(i, j + 1)].reverse().find(v => v != null);
  if (a != null && b != null){ const c = b / a - 1; $("tChg").textContent = fmtPct(c); $("tChg").style.color = c >= 0 ? "var(--green)" : "var(--red)"; }
}
function onTopRelayout(e){
  if (syncing) return;
  let a, b;
  if (e["xaxis.range[0]"]){ a = e["xaxis.range[0]"]; b = e["xaxis.range[1]"]; }
  else if (e["xaxis.range"]){ [a, b] = e["xaxis.range"]; }
  else if (e["xaxis.autorange"]){ apply(0, N - 1, "control"); return; }
  else return;
  let i = idxOf(String(a).slice(0, 10), "lo"), j = idxOf(String(b).slice(0, 10), "hi");
  if (j - i < 5) return;
  apply(i, j, "top");
}

// ---------------- 3D chart ----------------
function render3D(){
  const [i0, i1] = range, count = i1 - i0 + 1, step = Math.max(1, Math.ceil(count / MAX_PTS));
  const idx = []; for (let i = i0; i <= i1; i += step) idx.push(i); if (idx[idx.length - 1] !== i1) idx.push(i1);
  const dl = idx.map(i => D.dates[i]);

  let lines = D.series.map((s, n) => ({s, n})).filter(o => show.sec || o.s.kind === "index" || o.n === tk).map(({s, n}) => {
    let base = null; const r = [];
    for (let i = i0; i <= i1; i++){ const p = s.p[i]; if (p != null){ base = p; break; } }
    for (const i of idx){ const p = s.p[i]; r.push(p == null || base == null ? null : p / base - 1); }
    const last = [...r].reverse().find(v => v != null);
    return {name: s.name, kind: s.kind, n, r, last: last == null ? null : last};
  }).filter(l => l.last != null);
  lines.sort((a, b) => a.last - b.last);

  const all = lines.flatMap(l => l.r.filter(v => v != null)).map(Math.abs).sort((a, b) => a - b);
  const lim = Math.max(all[Math.floor(0.98 * (all.length - 1))] || 0.01, 0.01);
  let zmin = 0, zmax = 0; lines.forEach(l => l.r.forEach(v => { if (v != null){ zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); }}));
  const padz = (zmax - zmin) * 0.08 || 0.02; zmin -= padz; zmax += padz;
  const xs = idx.map((_, c) => c), X1 = xs.length - 1, nL = lines.length, yFront = -1.6, yBack = nL - 0.4;

  const traces = [];
  if (show.plane) traces.push({type: "surface", x: [0, X1], y: [yFront, yBack], z: [[0, 0], [0, 0]], showscale: false,
    opacity: 0.18, colorscale: [[0, "#aab6c4"], [1, "#aab6c4"]], hoverinfo: "skip", lighting: {ambient: 1, diffuse: 0, specular: 0}});
  D.crashes.forEach(cr => {
    if (cr.date < D.dates[i0] || cr.date > D.dates[i1]) return;
    let c = 0; while (c < X1 && dl[c + 1] <= cr.date) c++;
    traces.push({type: "mesh3d", x: [c, c, c, c], y: [yFront, yBack, yBack, yFront], z: [zmin, zmin, zmax, zmax],
      i: [0, 0], j: [1, 2], k: [2, 3], color: "#ffc83d", opacity: 0.09, hoverinfo: "skip", flatshading: true});
    traces.push({type: "scatter3d", mode: "lines+text", x: [c, c], y: [yBack, yBack], z: [zmin, zmax],
      line: {color: "#ffc83d", width: 2}, text: ["", cr.name], textposition: "top center",
      textfont: {color: "#ffc83d", size: 11}, hovertemplate: cr.name + "<extra></extra>"});
  });
  if (show.tda){
    let lmax = 0; idx.forEach(i => lmax = Math.max(lmax, D.l1[i]));
    traces.push({type: "scatter3d", mode: "lines", x: xs, y: xs.map(() => yFront), z: idx.map(i => lmax ? D.l1[i] / lmax * zmax * 0.9 : 0),
      line: {color: "#29d3ff", width: 4}, customdata: idx.map(i => [D.dates[i], D.l1[i]]),
      hovertemplate: "<b>TDA signal</b><br>%{customdata[0]}<br>L1 = %{customdata[1]:.3e}<extra></extra>"});
  }
  lines.forEach((l, k) => {
    const sel = l.n === tk;
    traces.push({type: "scatter3d", mode: "lines", x: xs, y: xs.map(() => k), z: l.r, connectgaps: false,
      line: {width: sel ? 8 : l.kind === "index" ? 4 : 2.5, color: l.r.map(v => v == null ? 0 : v), cmin: -lim, cmax: lim, colorscale: RG},
      customdata: dl, hovertemplate: `<b>${l.name}</b><br>%{customdata}<br>%{z:+.1%}<extra></extra>`});
  });
  // labels: selected ticker, top 3 and bottom 3 only (full list is in the side panel)
  const keep = new Set([0, 1, 2, nL - 1, nL - 2, nL - 3].filter(k => k >= 0 && k < nL));
  lines.forEach((l, k) => { if (l.n === tk) keep.add(k); });
  const lk = [...keep].sort((a, b) => a - b);
  traces.push({type: "scatter3d", mode: "text", x: lk.map(() => X1 * 1.01), y: lk, z: lk.map(k => lines[k].last),
    text: lk.map(k => `  ${short(lines[k].name)} ${fmtPct(lines[k].last)}`), textposition: "middle right",
    textfont: {size: 11, color: lk.map(k => lines[k].n === tk ? "#ffffff" : rgAt(lines[k].last, lim))}, hoverinfo: "skip"});

  const span = (toDate(D.dates[i1]) - toDate(D.dates[i0])) / 864e5, nt = 6, tv = [], tt = [];
  const f = span > 4*365 ? s => s.slice(0, 4) : span > 90 ? s => monthLabel(s) : s => `${monthLabel(s).slice(0,3)} ${+s.slice(8)}`;
  for (let t = 0; t < nt; t++){ const c = Math.round(t * X1 / (nt - 1)); tv.push(c); tt.push(f(dl[c])); }
  const ax = {backgroundcolor: "#0b0e14", showbackground: true, gridcolor: "#1b2330", zerolinecolor: "#1b2330",
              tickfont: {size: 11, color: "#7c8a99"}, title: {text: "", font: {size: 12, color: "#7c8a99"}}, showspikes: false};
  Plotly.react("chart", traces, {
    paper_bgcolor: "#0b0e14", margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false, uirevision: "cam",
    hoverlabel: {bgcolor: "#121722", bordercolor: "#2a3442", font: {color: "#e8ecf1"}},
    scene: {uirevision: "cam", camera: CAM, aspectmode: "manual", aspectratio: {x: 2.6, y: 1.1, z: 0.85},
      xaxis: {...ax, tickvals: tv, ticktext: tt, range: [0, X1 * 1.35]},
      yaxis: {...ax, tickvals: [], range: [yFront - 0.6, nL]},
      zaxis: {...ax, title: {...ax.title, text: "Return"}, tickformat: "+.0%", range: [zmin, zmax], nticks: 6}},
  }, {displaylogo: false, responsive: true, plotGlPixelRatio: Math.max(2, window.devicePixelRatio || 1)});
  updatePanel(lines, i0, i1);
}

// ---------------- side panel ----------------
function updatePanel(lines, i0, i1){
  const s = D.series[tk]; let base = null, peak = -Infinity, dd = 0, endv = null;
  for (let i = i0; i <= i1; i++){ const p = s.p[i]; if (p == null) continue; if (base == null) base = p; peak = Math.max(peak, p); dd = Math.min(dd, p / peak - 1); endv = p; }
  let bi = i0; for (let i = i0; i <= i1; i++) if (D.l1[i] > D.l1[bi]) bi = i;
  $("sFrom").textContent = D.dates[i0]; $("sTo").textContent = D.dates[i1];
  $("sDays").textContent = (i1 - i0 + 1).toLocaleString();
  $("sTk").textContent = short(s.name);
  if (base != null){ const c = endv / base - 1; $("sChg").textContent = fmtPct(c); $("sChg").style.color = c >= 0 ? "var(--green)" : "var(--red)"; }
  $("sDD").textContent = (dd * 100).toFixed(1) + "%"; $("sDD").style.color = "var(--red)";
  $("sPeak").textContent = D.dates[bi]; $("sPeak").style.color = "var(--cyan)";
  $("ranks").innerHTML = [...lines].reverse().map(l =>
    `<div class="rank${l.n === tk ? " sel" : ""}" data-n="${l.n}"><span>${short(l.name)}</span><span style="color:${l.last >= 0 ? "var(--green)" : "var(--red)"}">${fmtPct(l.last)}</span></div>`).join("");
  document.querySelectorAll(".rank").forEach(el => el.onclick = () => pickTicker(+el.dataset.n));
}
function pickTicker(n){ tk = n; $("ticker").value = n; initTop(); apply(range[0], range[1], "control"); }

// ---------------- controls ----------------
D.series.forEach((s, n) => { const o = document.createElement("option"); o.value = n; o.textContent = s.name; $("ticker").appendChild(o); });
$("ticker").onchange = e => pickTicker(+e.target.value);
[["1M",1],["3M",3],["6M",6],["1Y",12],["5Y",60],["All",0]].forEach(([label, months]) => {
  const b = document.createElement("button"); b.textContent = label;
  b.onclick = () => {
    if (!months){ apply(0, N - 1, "control"); return; }
    const s = toDate(D.dates[range[1]]); s.setUTCMonth(s.getUTCMonth() - months); setRange(iso(s), D.dates[range[1]]);
  };
  $("quick").appendChild(b);
});
D.crashes.forEach(c => {
  const b = document.createElement("button"); b.className = "chip"; b.innerHTML = `<span class="dot"></span>${c.name}`;
  b.onclick = () => { const s = toDate(c.date), e = toDate(c.date); s.setUTCMonth(s.getUTCMonth() - 12); e.setUTCMonth(e.getUTCMonth() + 4); setRange(iso(s), iso(e)); };
  $("crashBar").appendChild(b);
});
function search(){ const r = parseQuery($("q").value); if (!r){ $("err").textContent = "Try 2008, Mar 2020, or 2007-06 to 2009-03"; return; } setRange(r[0], r[1]); }
$("go").onclick = search;
$("q").addEventListener("keydown", e => { if (e.key === "Enter") search(); });
$("apply").onclick = () => setRange($("from").value, $("to").value);
$("back").onclick = () => shift(-1); $("fwd").onclick = () => shift(1);
document.addEventListener("keydown", e => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowLeft") shift(-1); if (e.key === "ArrowRight") shift(1);
});
const toggle = (id, key) => $(id).onclick = () => { show[key] = !show[key]; $(id).classList.toggle("on", show[key]); render3D(); };
toggle("tSec", "sec"); toggle("tTda", "tda"); toggle("tPlane", "plane");
$("resetCam").onclick = () => Plotly.relayout("chart", {"scene.camera": CAM});
$("from").min = $("to").min = D.dates[0]; $("from").max = $("to").max = D.dates[N - 1];

// ---------------- start ----------------
const h = decodeURIComponent(location.hash.slice(1)), hm = h.match(/^(\d{4}-\d\d-\d\d)\.\.(\d{4}-\d\d-\d\d)(?:&(.*))?/);
if (hm && hm[3]){ const n = D.series.findIndex(s => s.name === hm[3]); if (n >= 0) tk = n; }
$("ticker").value = tk;
initTop();
document.getElementById("top").on("plotly_relayout", onTopRelayout);
if (hm) setRange(hm[1], hm[2]);
else if (D.init.start || D.init.end) setRange(D.init.start || D.dates[0], D.init.end || D.dates[N - 1]);
else { const s = toDate(D.dates[N - 1]); s.setUTCFullYear(s.getUTCFullYear() - 2); setRange(iso(s), D.dates[N - 1]); }
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="initial start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="initial end date YYYY-MM-DD")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prices = load_prices()
    sectors = load_sectors(str(prices.index[0].date()))
    norms = load_norms(prices)
    payload = build_payload(prices, sectors, norms, args.start, args.end)

    html = PAGE.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    path = os.path.join(OUT_DIR, OUT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(path))


if __name__ == "__main__":
    main()
