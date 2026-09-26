// The tile layout explorer on the Programming model page.

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
