// The filter on the Language builtins and Host API pages. The layout plugin renders
// the entries at build time; this script only hides the ones that don't match.
var input = document.getElementById("ref-search");
var countEl = document.getElementById("ref-count");
var empty = document.getElementById("ref-empty");
var groups = Array.prototype.slice.call(document.querySelectorAll("#ref-list .refgroup"));
var entries = groups.map(function (g) {
  var title = g.querySelector("h2").textContent;
  return Array.prototype.slice.call(g.querySelectorAll(".entry")).map(function (el) {
    return { el: el, html: el.innerHTML, text: (el.textContent + " " + title).toLowerCase() };
  });
});
var total = entries.reduce(function (n, g) { return n + g.length; }, 0);

function mark(html, q) {
  if (!q) return html;
  var re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
  return html.replace(/(<[^>]+>)|([^<]+)/g, function (m, tag, text) { return tag ? tag : text.replace(re, "<mark>$1</mark>"); });
}

function filter() {
  var q = input.value.trim().toLowerCase();
  var shown = 0;
  groups.forEach(function (g, i) {
    var visible = 0;
    entries[i].forEach(function (e) {
      var on = !q || e.text.indexOf(q) !== -1;
      e.el.hidden = !on;
      e.el.innerHTML = on ? mark(e.html, q) : e.html;
      if (on) visible++;
    });
    g.hidden = visible === 0;
    shown += visible;
  });
  empty.hidden = shown !== 0;
  countEl.textContent = shown + " of " + total;
}

input.addEventListener("input", filter);
filter();
