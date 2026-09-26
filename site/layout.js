// A Vite plugin that wraps each page in the shared layout at build time.
//
// A page file holds only its own content: an optional hero before <main>, the
// <main> element, and any page-specific <script type="module"> tags. The plugin
// adds the document head, the header, the sidebar with the page's section links,
// the previous and next links, and the footer, so every page is static HTML.
import { HEADER_LINKS, NOT_FOUND, PAGES } from "./pages.js";

const SITE = "Enceladus";
const REPO = "https://github.com/kashifulhaque/enceladus";
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

function sidebar(page, toc) {
  const groups = [];
  for (const p of PAGES) {
    let g = groups.find((x) => x.name === p.group);
    if (!g) groups.push((g = { name: p.group, pages: [] }));
    g.pages.push(p);
  }
  const links = groups
    .map((g) => {
      const items = g.pages
        .map((p) => {
          if (p !== page) return `<a href="${p.url}">${esc(p.title)}</a>`;
          const sections = toc.length > 1
            ? `<div class="toc">${toc.map((t) => `<a href="#${t.id}">${t.text}</a>`).join("")}</div>`
            : "";
          return `<a class="current" href="${p.url}" aria-current="page">${esc(p.title)}</a>${sections}`;
        })
        .join("\n      ");
      return `    <h4>${esc(g.name)}</h4>\n      ${items}`;
    })
    .join("\n");
  const mobile = PAGES.map((p) =>
    p === page
      ? `<a href="${p.url}" aria-current="page">${esc(p.title)}</a>`
      : `<a href="${p.url}">${esc(p.title)}</a>`,
  ).join("");
  return {
    side: `<nav class="sidenav" aria-label="Documentation">\n${links}\n  </nav>`,
    mobile: `<details class="mobnav">\n  <summary>${esc(page.title)}<span>All pages</span></summary>\n  <div>${mobile}</div>\n</details>`,
  };
}

function pager(page) {
  const i = PAGES.indexOf(page);
  if (i < 0) return "";
  const prev = PAGES[i - 1];
  const next = PAGES[i + 1];
  const link = (p, dir) =>
    `<a class="${dir}" href="${p.url}"><span>${dir === "prev" ? "Previous" : "Next"}</span>${esc(p.title)}</a>`;
  return `<nav class="pager" aria-label="Previous and next pages">${prev ? link(prev, "prev") : "<span></span>"}${next ? link(next, "next") : ""}</nav>`;
}

function pageCards() {
  const cards = PAGES.slice(1)
    .map(
      (p) =>
        `<a class="card" href="${p.url}"><span class="k">${esc(p.group)}</span><b>${esc(p.title)}</b><span>${esc(p.description)}</span></a>`,
    )
    .join("\n    ");
  return `<div class="cards">\n    ${cards}\n  </div>`;
}

function header(page) {
  const nav = HEADER_LINKS.map((url) => {
    const p = PAGES.find((x) => x.url === url);
    const current = p === page ? ' aria-current="page"' : "";
    return `<a class="hide-sm" href="${p.url}"${current}>${esc(p.title.replace("Language reference", "Reference"))}</a>`;
  }).join("\n      ");
  return `<header class="topbar">
  <div class="wrap">
    <a class="brand" href="/" aria-label="${SITE} docs home">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
        <circle cx="13" cy="13" r="11.5" fill="none" stroke="currentColor" stroke-width="1.5"/>
        <path d="M6.5 16.5c2.2-1.2 4.6-1.6 7.2-1.1M8 19.4c2.4-1 5-1.2 7.6-.4M9.8 13.4c2-.9 4.2-1.1 6.4-.6" stroke="var(--accent)" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      </svg>
      <b>${SITE}</b>
    </a>
    <span class="chip">v0.1.0 · alpha</span>
    <nav class="topnav" aria-label="Primary">
      ${nav}
      <a href="${REPO}">GitHub</a>
    </nav>
  </div>
</header>`;
}

function render(page, source) {
  const scripts = (source.match(/<script\b[^>]*>[\s\S]*?<\/script>/g) || []).join("\n");
  const content = source.replace(/<script\b[^>]*>[\s\S]*?<\/script>\s*/g, "");
  const start = content.indexOf("<main");
  const end = content.lastIndexOf("</main>");
  if (start < 0 || end < 0) throw new Error(`${page.file}: the page must have a <main> element`);
  const hero = content.slice(0, start).trim();
  const { html: main, toc } = addHeadingIds(content.slice(start, end + "</main>".length));
  const nav = sidebar(page, toc);
  const isHome = page.url === "/";
  const title = isHome ? `${SITE} docs` : `${page.title} · ${SITE}`;
  const body = main
    .replace("<!-- @page-cards -->", pageCards())
    .replace(/<main([^>]*)>/, `<main$1>\n${nav.mobile}`)
    .replace(/<\/main>$/, `${pager(page)}\n</main>`);

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>${esc(title)}</title>
  <meta name="description" content="${esc(page.description)}">
  <meta property="og:title" content="${esc(title)}">
  <meta property="og:description" content="${esc(page.description)}">
  <meta property="og:type" content="website">
  <meta name="color-scheme" content="light dark">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="${FONTS}">
  <script type="module" src="/src/main.js"></script>
</head>
<body${isHome ? ' class="home"' : ""}>
${header(page)}
${hero}
<div class="wrap docs">
  ${nav.side}
${body}
</div>
<footer>
  <div class="wrap">
    <span>${SITE} 0.1.0 · Apache-2.0</span>
    <span><a href="${REPO}">github.com/kashifulhaque/enceladus</a></span>
  </div>
</footer>
${scripts}
</body>
</html>
`;
}

export function layout() {
  const pages = [...PAGES, NOT_FOUND];
  return {
    name: "enceladus-layout",
    transformIndexHtml: {
      order: "pre",
      handler(html, ctx) {
        const file = ctx.path.replace(/^\//, "");
        const page = pages.find((p) => p.file === file);
        if (!page) throw new Error(`${file} isn't listed in site/pages.js`);
        return render(page, html);
      },
    },
  };
}
