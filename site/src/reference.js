// The filterable language reference on the Language reference page.
import { REFERENCE } from "./data/reference.js";

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
