// A Vite plugin that wraps each page in the shared layout at build time.
//
// A page file holds only its own content: an optional hero before <main>, the
// <main> element, and any page-specific <script type="module"> tags. The plugin
// adds the document head, the header, the section's sidebar with the page's
// section links, the previous and next links, and the footer, so every page is
// static HTML. The home page gets the header and footer but no sidebar.
//
// Pages can use the following directives, which the plugin expands first:
//
//   <!-- @sections -->              a card for each section, for the home page
//   <!-- @cards SECTION -->         a card for each page in SECTION except its first
//   <!-- @source PATH -->           a code block with the file at PATH, relative to
//                                   the repository root; PATH#L10-L40 shows lines 10-40
//   <!-- @results NAME -->          the output of benchmark NAME from RESULTS_FILE
//   <!-- @reference NAMESPACE -->   the reference entries for "tl" or "host"
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { REFERENCE } from "./src/data/reference.js";
import { HOME, NOT_FOUND, PAGES, SECTIONS } from "./pages.js";

const SITE = "Enceladus";
const REPO = "https://github.com/kashifulhaque/enceladus";
const ROOT = resolve(import.meta.dirname, "..");
export const RESULTS_FILE = "benchmarks/results/2026-09-27-applegpu_g16s-2.md";
const FONTS =
  "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,650;12..96,800" +
  "&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap";

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const stripTags = (s) => s.replace(/<[^>]+>/g, "");
const slug = (s) =>
  stripTags(s)
    .toLowerCase()
    .replace(/&[a-z]+;|&#\d+;|['’]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

// Gives every <h2> in the page an id, and returns the page's section list.
function addHeadingIds(main) {
  const toc = [];
  const used = new Set();
  const html = main.replace(/<h2(\s[^>]*)?>([\s\S]*?)<\/h2>/g, (whole, attrs = "", inner) => {
    const existing = attrs.match(/\sid="([^"]+)"/);
    let id = existing ? existing[1] : slug(inner);
    if (!existing) {
      for (let n = 2; used.has(id); n++) id = `${slug(inner)}-${n}`;
    }
    used.add(id);
    toc.push({ id, text: inner });
    return existing ? whole : `<h2 id="${id}"${attrs}>${inner}</h2>`;
  });
  return { html, toc };
}

/* Directives */

function card(p, kicker) {
  return `<a class="card" href="${p.url}"><span class="k">${esc(kicker)}</span><b>${esc(p.title)}</b><span>${esc(p.description)}</span></a>`;
}

function sectionCards() {
  const cards = SECTIONS.map((s) => {
    const n = s.pages.length;
    const count = s.id === "examples" ? `${n - 1} examples` : `${n} ${n === 1 ? "page" : "pages"}`;
    return card({ url: s.pages[0].url, title: s.title, description: s.description }, count);
  });
  return `<div class="cards doors">\n    ${cards.join("\n    ")}\n  </div>`;
}

function pageCards(id) {
  const section = SECTIONS.find((s) => s.id === id);
  if (!section) throw new Error(`@cards: no section named ${id}`);
  const cards = section.pages.slice(1).map((p) => card(p, p.group));
  return `<div class="cards">\n    ${cards.join("\n    ")}\n  </div>`;
}

function readRepoFile(path) {
  try {
    return readFileSync(resolve(ROOT, path), "utf8");
  } catch {
    throw new Error(`@source: can't read ${path} from the repository root`);
  }
}

function source(spec) {
  const [path, range] = spec.split("#");
  let lines = readRepoFile(path).replace(/\s+$/, "").split("\n");
  let from = 1;
  if (range) {
    const m = range.match(/^L(\d+)-L(\d+)$/);
    if (!m) throw new Error(`@source: write the range as #L10-L40, not #${range}`);
    from = Number(m[1]);
    lines = lines.slice(from - 1, Number(m[2]));
  }
  const lang = path.endsWith(".py") ? "python" : path.endsWith(".metal") ? "cpp" : "bash";
  const anchor = range ? `#L${from}-L${from + lines.length - 1}` : "";
  return `<div class="codeblock"><div class="bar"><span class="file">${esc(path)}</span><span class="spacer"></span>` +
    `<a class="copy" href="${REPO}/blob/main/${path}${anchor}">GitHub</a><button class="copy" type="button">Copy</button></div>` +
    `<pre><code class="language-${lang}">${esc(lines.join("\n"))}</code></pre></div>`;
}

function results(name) {
  const text = readRepoFile(RESULTS_FILE);
  const m = text.match(new RegExp(`## ${name}\\n\\n\`\`\`text\\n([\\s\\S]*?)\\n\`\`\``));
  if (!m) throw new Error(`@results: ${RESULTS_FILE} has no ${name} section`);
  return `<div class="codeblock"><div class="bar"><span class="file">${esc(name)} output</span><span class="spacer"></span>` +
    `<a class="copy" href="${REPO}/blob/main/${RESULTS_FILE}">Full report</a></div>` +
    `<pre class="text"><code class="nohighlight">${esc(m[1])}</code></pre></div>`;
}

function sigHTML(sig, ns) {
  const prefix = ns === "tl" && !/^(desc\.|x\.)/.test(sig) ? "tl." : ns === "host" && !/^(kernel\.|Tensor\.)/.test(sig) ? "enceladus." : "";
  return sig
    .split(/,\s(?=[a-z_]+\()/)
    .map((part) => {
      const m = part.match(/^([\w.]+)(\(.*\))?$/);
      if (!m) return esc(part);
      return `<span class="pa">${prefix}</span><span class="nm">${esc(m[1])}</span>` + (m[2] ? `<span class="pa">${esc(m[2])}</span>` : "");
    })
    .join('<span class="pa">, </span>');
}

function reference(ns) {
  const groups = REFERENCE.filter((g) => g[0] === ns);
  if (!groups.length) throw new Error(`@reference: no entries for ${ns}`);
  return groups
    .map(([, title, note, items]) =>
      `<div class="refgroup"><h2>${esc(title)}</h2>` +
      (note ? `<p>${esc(note)}</p>` : "") +
      `<div class="entries">` +
      items.map(([sig, desc]) => `<div class="entry"><div class="sig">${sigHTML(sig, ns)}</div><p>${esc(desc)}</p></div>`).join("") +
      `</div></div>`,
    )
    .join("\n");
}

function expand(html) {
  return html.replace(/<!--\s*@(\w+)\s*([^\s]*)\s*-->/g, (whole, name, arg) => {
    if (name === "sections") return sectionCards();
    if (name === "cards") return pageCards(arg);
    if (name === "source") return source(arg);
    if (name === "results") return results(arg);
    if (name === "reference") return reference(arg);
    throw new Error(`unknown directive @${name}`);
  });
}

/* Navigation */

function sidebar(page, toc) {
  const section = page.section;
  const groups = [];
  for (const p of section.pages) {
    let g = groups.find((x) => x.name === p.group);
    if (!g) groups.push((g = { name: p.group, pages: [] }));
    g.pages.push(p);
  }
  const links = groups
    .map((g) => {
      const items = g.pages
        .map((p) => {
          if (p.url !== page.url) return `<a href="${p.url}">${esc(p.title)}</a>`;
          const sections = toc.length > 1
            ? `<div class="toc">${toc.map((t) => `<a href="#${t.id}">${t.text}</a>`).join("")}</div>`
            : "";
          return `<a class="current" href="${p.url}" aria-current="page">${esc(p.title)}</a>${sections}`;
        })
        .join("\n      ");
      return `    <h4>${esc(g.name)}</h4>\n      ${items}`;
    })
    .join("\n");
  const mobile = section.pages
    .map((p) => `<a href="${p.url}"${p.url === page.url ? ' aria-current="page"' : ""}>${esc(p.title)}</a>`)
    .join("");
  return {
    side: `<nav class="sidenav" aria-label="${esc(section.title)}">\n${links}\n  </nav>`,
    mobile: `<details class="mobnav">\n  <summary>${esc(page.title)}<span>${esc(section.title)} pages</span></summary>\n  <div>${mobile}</div>\n</details>`,
  };
}

// Links to the previous and next pages in the same section.
function pager(page) {
  const pages = page.section.pages;
  const i = pages.findIndex((p) => p.url === page.url);
  const prev = pages[i - 1];
  const next = pages[i + 1];
  if (!prev && !next) return "";
  const link = (p, dir) =>
    `<a class="${dir}" href="${p.url}"><span>${dir === "prev" ? "Previous" : "Next"}</span>${esc(p.title)}</a>`;
  return `<nav class="pager" aria-label="Previous and next pages">${prev ? link(prev, "prev") : "<span></span>"}${next ? link(next, "next") : ""}</nav>`;
}

function header(page) {
  const nav = SECTIONS.map((s) => {
    const current = page.section === s ? ' aria-current="true"' : "";
    return `<a href="${s.pages[0].url}"${current}>${esc(s.title)}</a>`;
  }).join("\n      ");
  return `<header class="topbar">
  <div class="wrap">
    <a class="brand" href="/" aria-label="${SITE} home">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
        <circle cx="13" cy="13" r="11.5" fill="none" stroke="currentColor" stroke-width="1.5"/>
        <path d="M6.5 16.5c2.2-1.2 4.6-1.6 7.2-1.1M8 19.4c2.4-1 5-1.2 7.6-.4M9.8 13.4c2-.9 4.2-1.1 6.4-.6" stroke="var(--accent)" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      </svg>
      <b>${SITE}</b>
    </a>
    <span class="chip">v0.1.0 · alpha</span>
    <nav class="topnav" aria-label="Primary">
      ${nav}
      <a class="gh" href="${REPO}" aria-label="GitHub repository">
        <svg width="18" height="18" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      </a>
    </nav>
  </div>
</header>`;
}

function head(title, description) {
  return `<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>${esc(title)}</title>
  <meta name="description" content="${esc(description)}">
  <meta property="og:title" content="${esc(title)}">
  <meta property="og:description" content="${esc(description)}">
  <meta property="og:type" content="website">
  <meta name="color-scheme" content="light dark">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="${FONTS}">
  <script type="module" src="/src/main.js"></script>
</head>`;
}

const FOOTER = `<footer>
  <div class="wrap">
    <span>${SITE} 0.1.0 · Apache-2.0</span>
    <span><a href="${REPO}">github.com/kashifulhaque/enceladus</a></span>
  </div>
</footer>`;

function render(page, source) {
  const scripts = (source.match(/<script\b[^>]*>[\s\S]*?<\/script>/g) || []).join("\n");
  const content = expand(source.replace(/<script\b[^>]*>[\s\S]*?<\/script>\s*/g, ""));
  const start = content.indexOf("<main");
  const end = content.lastIndexOf("</main>");
  if (start < 0 || end < 0) throw new Error(`${page.file}: the page must have a <main> element`);
  const hero = content.slice(0, start).trim();
  const main = content.slice(start, end + "</main>".length);

  // The home page and the 404 page have no sidebar.
  if (!page.section) {
    const isHome = page === HOME;
    const title = isHome ? `${SITE}: GPU kernels in Python for Apple silicon` : `${page.title} · ${SITE}`;
    return `<!doctype html>
<html lang="en">
${head(title, page.description)}
<body class="${isHome ? "home" : "bare"}">
${header(page)}
${hero}
<div class="wrap">
${main}
</div>
${FOOTER}
${scripts}
</body>
</html>
`;
  }

  const { html, toc } = addHeadingIds(main);
  const nav = sidebar(page, toc);
  const title = `${page.title} · ${page.section.title} · ${SITE}`;
  const body = html
    .replace(/<main([^>]*)>/, `<main$1>\n${nav.mobile}`)
    .replace(/<\/main>$/, `${pager(page)}\n</main>`);
  return `<!doctype html>
<html lang="en">
${head(title, page.description)}
<body>
${header(page)}
${hero}
<div class="wrap docs">
  ${nav.side}
${body}
</div>
${FOOTER}
${scripts}
</body>
</html>
`;
}

export const ALL_PAGES = [HOME, ...PAGES, NOT_FOUND];

export function layout() {
  return {
    name: "enceladus-layout",
    transformIndexHtml: {
      order: "pre",
      handler(html, ctx) {
        const file = ctx.path.replace(/^\//, "");
        const page = ALL_PAGES.find((p) => p.file === file);
        if (!page) throw new Error(`${file} isn't listed in site/pages.js`);
        return render(page, html);
      },
    },
  };
}
