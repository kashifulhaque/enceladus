// The throughput chart on the Benchmarks page.
import { BENCHMARKS } from "./data/benchmarks.js";

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
