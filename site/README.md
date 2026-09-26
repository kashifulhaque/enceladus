# Enceladus docs site

This directory holds the single-page documentation site for Enceladus. It's a static
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

The page content is in the following files:

- `index.html`: the prose, code samples, and tables.
- `src/data/reference.js`: the language reference entries. Update them when a builtin
  changes in `docs/guide/language-reference.md`.
- `src/data/benchmarks.js`: the benchmark rows, copied from `benchmarks/results/`.
- `src/main.js`: the tabs, copy buttons, tile layout explorer, reference filter, and
  benchmark chart.
- `src/style.css`: the design tokens for light and dark mode, and the layout.

The generated MSL in the page's first code panel comes from
`add_kernel.warmup(x, y, out, n, BLOCK=1024).msl`. When code generation changes,
regenerate it and paste the kernel's entry point into `index.html`.
