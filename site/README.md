# Enceladus docs site

This directory holds the documentation site for Enceladus. It's a static, multi-page
[Vite](https://vite.dev/) project with no framework, and it builds to `dist/`.

## Develop locally

To start a dev server with hot reload, run the following commands from this directory:

```bash
npm install
npm run dev
```

To build the site and serve the production build, run `npm run build` and then
`npm run preview`.

## Deploy to Cloudflare Pages

To deploy from Git, create a Pages project connected to this repository with the
following build settings:

| Setting | Value |
|---|---|
| Framework preset | Vite |
| Root directory | `site` |
| Build command | `npm run build` |
| Build output directory | `dist` |

Cloudflare reads the Node.js version from `.node-version`. Vite 8 needs Node.js 20.19
or later, or 22.12 or later.

To deploy a local build without Git, run the following command from this directory:

```bash
npx wrangler pages deploy dist --project-name enceladus-docs
```

## Update the content

Each page is an HTML file that holds only its own content: an optional hero, a
`<main>` element, and any page-specific scripts. At build time, the layout plugin in
`layout.js` wraps every page in the shared header, sidebar, previous and next links,
and footer. It also gives each `<h2>` an ID and lists it in the sidebar under the
current page.

The site has the following files:

- `pages.js`: the page list, in reading order, with each page's title, sidebar group,
  and description. To add a page, create `NAME/index.html` and add an entry here.
- `index.html` and `*/index.html`: the page content.
- `404.html`: the page that Cloudflare Pages serves for unknown URLs.
- `src/main.js`: highlighting, copy buttons, tabs, and the sidebar's scroll tracking,
  which every page loads.
- `src/explorer.js`, `src/reference.js`, and `src/benchmarks.js`: the tile layout
  explorer, the reference filter, and the benchmark chart, each loaded by one page.
- `src/data/reference.js`: the language reference entries. Update them when a builtin
  changes in `docs/guide/language-reference.md`.
- `src/data/benchmarks.js`: the benchmark rows, copied from `benchmarks/results/`.
- `src/style.css`: the design tokens for light and dark mode, and the layout.

The generated MSL in the overview's first code panel comes from
`add_kernel.warmup(x, y, out, n, BLOCK=1024).msl`. When code generation changes,
regenerate it and paste the kernel's entry point into `index.html`.
