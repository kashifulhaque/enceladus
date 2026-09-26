import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import cpp from "highlight.js/lib/languages/cpp";
import python from "highlight.js/lib/languages/python";

import { BENCHMARKS } from "./data/benchmarks.js";
import { REFERENCE } from "./data/reference.js";
import "./style.css";

/* Syntax highlighting */
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("cpp", cpp);
hljs.registerLanguage("python", python);
document.querySelectorAll("pre code[class*='language-']").forEach((el) => hljs.highlightElement(el));

/* Copy buttons */
function copyText(text, btn) {
  function done(ok) {
    var old = btn.dataset.label || btn.textContent;
    btn.dataset.label = old;
    btn.textContent = ok ? "Copied" : "Select to copy";
    setTimeout(function () { btn.textContent = old; }, 1400);
  }
  try {
    navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
  } catch (e) { done(false); }
}
document.querySelectorAll(".copy").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var target = btn.dataset.copy ? document.querySelector(btn.dataset.copy) : null;
    if (!target) {
      var block = btn.closest(".codeblock");
      target = block.querySelector("pre:not([hidden]) code");
    }
    copyText(target.textContent, btn);
  });
});

/* Tabs */
var captions = {
  "qs-add": "The grid launches <code>cdiv(98_432, 1024) = 97</code> programs. The last one covers only 128 valid elements, and <code>mask</code> keeps it in bounds.",
  "qs-softmax": "Tile dimensions must be powers of two, so 1,000-column rows use a 1,024-column block. Masked lanes load <code>-inf</code>, which leaves the max unchanged and adds 0 to the sum.",
  "qs-matmul": "<code>tl.dot</code> reads descriptor loads straight from device memory and accumulates in <code>float32</code>. Descriptors handle the ragged 1,000-row edge without masks.",
  "qs-attn": "Excerpt from <code>examples/08_flash_attention.py</code>. The full example also has an autotuned version and a float64 NumPy reference."
};
document.querySelectorAll("[data-tabs]").forEach(function (block) {
  var tabs = block.querySelectorAll(".tab");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        document.getElementById(t.dataset.panel).hidden = !on;
      });
      var cap = block.querySelector("#qs-caption");
      if (cap && captions[tab.dataset.panel]) cap.innerHTML = captions[tab.dataset.panel];
    });
  });
});

/* Tile layout explorer. Mirrors the generated MSL:
   tc = (lane << 2) | (warp << 7); registers 0-3 hold tc..tc+3, registers 4-7 hold tc+512..tc+515. */
var N = 98432, BLOCK = 1024;
var grid = document.getElementById("tilegrid");
var cells = [];
var pid = 0, sel = 517;
function place(e) {
  var hi = e >= 512 ? 1 : 0;
  var base = e - hi * 512;
  var tc = base - (base & 3);
  return { lane: (tc >> 2) & 31, warp: (tc >> 7) & 3, reg: (base & 3) + hi * 4 };
}
var frag = document.createDocumentFragment();
for (var e = 0; e < BLOCK; e++) {
  var c = document.createElement("i");
  c.dataset.e = e;
  c.style.setProperty("--c", "var(--w" + place(e).warp + ")");
  frag.appendChild(c);
  cells.push(c);
}
grid.appendChild(frag);
function render() {
  var p = place(sel);
  var offs = pid * BLOCK + sel;
  cells.forEach(function (c, i) {
    c.classList.toggle("m", pid * BLOCK + i >= N);
    var q = place(i);
    c.classList.toggle("same", i !== sel && q.lane === p.lane && q.warp === p.warp);
    c.classList.toggle("sel", i === sel);
  });
  document.getElementById("r-offs").textContent = offs.toLocaleString("en-US");
  document.getElementById("r-warp").textContent = p.warp;
  document.getElementById("r-lane").textContent = p.lane;
  document.getElementById("r-reg").textContent = "x_0[" + p.reg + "]";
  var m = offs < N;
  document.getElementById("r-mask").textContent = m ? "true" : "false (" + offs.toLocaleString("en-US") + " >= n)";
  document.getElementById("r-msl").textContent = "thread " + (p.warp * 32 + p.lane) + " · ar_0[" + p.reg + "] = tc_0 + " + ((sel >= 512 ? 512 : 0) + (p.reg & 3));
}
function pick(e) {
  sel = Math.max(0, Math.min(BLOCK - 1, e | 0));
  document.getElementById("elem-input").value = sel;
  render();
}
grid.addEventListener("pointerover", function (ev) { if (ev.target.dataset.e) pick(+ev.target.dataset.e); });
grid.addEventListener("click", function (ev) { if (ev.target.dataset.e) pick(+ev.target.dataset.e); });
document.getElementById("elem-input").addEventListener("input", function (ev) {
  var v = parseInt(ev.target.value, 10);
  if (!isNaN(v)) { sel = Math.max(0, Math.min(BLOCK - 1, v)); render(); }
});
document.querySelectorAll("[data-pid]").forEach(function (b) {
  b.addEventListener("click", function () {
    pid = +b.dataset.pid;
    document.querySelectorAll("[data-pid]").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
    render();
  });
});
render();

/* Language reference */
var list = document.getElementById("ref-list");
var countEl = document.getElementById("ref-count");
var scope = "all";
var total = 0;
REFERENCE.forEach(function (g) { total += g[3].length; });
function esc(s) { return s.replace(/[&<>"]/g, function (ch) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]; }); }
function mark(html, q) {
  if (!q) return html;
  var re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
  return html.replace(/(<[^>]+>)|([^<]+)/g, function (m, tag, text) { return tag ? tag : text.replace(re, "<mark>$1</mark>"); });
}
function sigHTML(sig, ns) {
  var prefix = ns === "tl" && !/^(desc\.|x\.)/.test(sig) ? "tl." : (ns === "host" && !/^(kernel\.|Tensor\.)/.test(sig) ? "enceladus." : "");
  return sig.split(/,\s(?=[a-z_]+\()/).map(function (part) {
    var m = part.match(/^([\w.]+)(\(.*\))?$/);
    if (!m) return esc(part);
    return '<span class="pa">' + prefix + '</span><span class="nm">' + esc(m[1]) + '</span>' + (m[2] ? '<span class="pa">' + esc(m[2]) + '</span>' : "");
  }).join('<span class="pa">, </span>');
}
function renderRef() {
  var q = document.getElementById("ref-search").value.trim().toLowerCase();
  var html = "", shown = 0;
  REFERENCE.forEach(function (g) {
    if (scope !== "all" && scope !== g[0]) return;
    var items = g[3].filter(function (it) {
      return !q || (it[0] + " " + it[1] + " " + g[1]).toLowerCase().indexOf(q) !== -1;
    });
    if (!items.length) return;
    shown += items.length;
    html += '<div class="refgroup"><h3>' + esc(g[1]) + '<small>' + (g[0] === "tl" ? "tl" : "enceladus") + '</small></h3>' +
      (g[2] ? "<p>" + esc(g[2]) + "</p>" : "") + '<div class="entries">' +
      items.map(function (it) {
        return '<div class="entry"><div class="sig">' + mark(sigHTML(it[0], g[0]), q) + "</div><p>" + mark(esc(it[1]), q) + "</p></div>";
      }).join("") + "</div></div>";
  });
  list.innerHTML = html || '<div class="empty">No builtins match that filter. Try a shorter term, such as <code>load</code>.</div>';
  countEl.textContent = shown + " of " + total;
}
document.getElementById("ref-search").addEventListener("input", renderRef);
document.querySelectorAll("[data-scope]").forEach(function (b) {
  b.addEventListener("click", function () {
    scope = b.dataset.scope;
    document.querySelectorAll("[data-scope]").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
    renderRef();
  });
});
renderRef();

/* Benchmarks: Enceladus throughput / MLX throughput, minimum times. */
var LO = -20, HI = 60;
function pos(v) { return (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * 100; }
var out = '<div class="brow ticks" aria-hidden="true"><span></span><div class="track" style="height:14px">';
[-20, 0, 20, 40, 60].forEach(function (t) {
  out += '<span class="tick" style="left:' + pos(t) + '%">' + (t > 0 ? "+" : "") + t + '%</span>';
});
out += "</div><span></span></div>";
BENCHMARKS.forEach(function (g) {
  out += '<div class="bgroup">' + g[0] + "</div>";
  g[1].forEach(function (r) {
    var pct = (r[2] / r[3] - 1) * 100;
    var z = pos(0), p = pos(pct);
    var neg = pct < 0;
    var left = Math.min(z, p), width = Math.abs(p - z);
    var unit = g[0].indexOf("GB/s") !== -1 ? " GB/s" : " TFLOPS";
    var label = (pct >= 0 ? "+" : "−") + Math.abs(pct).toFixed(0) + "%";
    var desc = r[0] + ", " + r[1] + ": Enceladus " + r[2].toFixed(2) + unit + ", MLX " + r[3].toFixed(2) + unit + ", " + label;
    out += '<div class="brow" title="' + desc + '" aria-label="' + desc + '">' +
      '<div class="blabel">' + r[0] + '<small>' + r[1] + ' · ' + r[2].toFixed(2) + ' vs ' + r[3].toFixed(2) + '</small></div>' +
      '<div class="track">' +
      [0, 20, 40].map(function (t) { return t === 0 ? "" : '<span class="grid" style="left:' + pos(t) + '%"></span>'; }).join("") +
      '<span class="grid" style="left:' + pos(-20) + '%"></span><span class="grid" style="left:' + pos(60) + '%"></span>' +
      '<span class="bar' + (neg ? " neg" : "") + (pct > HI ? " clip" : "") + '" style="left:' + left + '%;width:' + width + '%;background:var(' + (neg ? "--neg" : "--pos") + ')"></span>' +
      '<span class="zero" style="left:calc(' + z + '% - .75px)"></span>' +
      "</div>" +
      '<div class="bval">' + label + "</div></div>";
  });
});
document.getElementById("bench-rows").innerHTML = out;

/* Scroll spy for the side navigation */
var links = Array.prototype.slice.call(document.querySelectorAll(".sidenav a"));
var targets = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
if ("IntersectionObserver" in window) {
  var visible = {};
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting; });
    var first = targets.find(function (t) { return t && visible[t.id]; });
    if (first) links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + first.id); });
  }, { rootMargin: "-80px 0px -55% 0px" });
  targets.forEach(function (t) { if (t) io.observe(t); });
}
