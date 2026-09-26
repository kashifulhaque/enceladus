import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import cpp from "highlight.js/lib/languages/cpp";
import python from "highlight.js/lib/languages/python";

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

/* Tabs. A tab's data-caption, if any, replaces the text of the block's [data-tab-caption]. */
document.querySelectorAll("[data-tabs]").forEach(function (block) {
  var tabs = block.querySelectorAll(".tab");
  var cap = block.querySelector("[data-tab-caption]");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        document.getElementById(t.dataset.panel).hidden = !on;
      });
      if (cap && tab.dataset.caption) cap.innerHTML = tab.dataset.caption;
    });
  });
});

/* Scroll spy: marks the sidebar link of the last section heading above the fold. */
var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".sidenav .toc a"));
var headings = tocLinks.map(function (a) { return document.getElementById(a.hash.slice(1)); });
function spy() {
  var current = 0;
  headings.forEach(function (h, i) { if (h && h.getBoundingClientRect().top < 120) current = i; });
  tocLinks.forEach(function (a, i) { a.classList.toggle("active", i === current); });
}
if (tocLinks.length) {
  var queued = false;
  window.addEventListener("scroll", function () {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; spy(); });
  }, { passive: true });
  spy();
}
