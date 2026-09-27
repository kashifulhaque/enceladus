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

The build reads files from outside `site/`, such as `examples/` and `benchmarks/`, so
it needs the whole repository. Cloudflare Pages clones the whole repository even when
the root directory is `site`.

Cloudflare reads the Node.js version from `.node-version`. Vite 8 needs Node.js 20.19
or later, or 22.12 or later.

To deploy a local build without Git, run the following command from this directory:

```bash
npx wrangler pages deploy dist --project-name enceladus-docs
```

## Site structure

The site has a home page and four sections. Each section has its own sidebar, and the
header links to the first page of each one.

| Section | URL | Content |
|---|---|---|
| Docs | `/docs/` | The introduction, the quickstart, the concepts, and the guides. |
| Reference | `/reference/` | The `tl` builtins, the host API, the data types, and the environment variables. |
| Examples | `/examples/` | One page for each program in the repository's `examples/` directory. |
| Benchmarks | `/benchmarks/` | The overview chart, one page for each benchmark, and the methodology. |

The home page holds only the hero and a card for each section. `public/_redirects`
sends the pre-section guide URLs, such as `/quickstart/`, to their pages under `/docs/`.

## Update the content

Each page is an HTML file that holds only its own content: an optional hero, a
`<main>` element, and any page-specific scripts. At build time, the layout plugin in
`layout.js` wraps every page in the shared header, the section's sidebar, the previous
and next links, and the footer. It also gives each `<h2>` an ID and lists it in the
sidebar under the current page.

To add a page, create `SECTION/NAME/index.html` and add an entry to that section's
`pages` list in `pages.js`.

Pages can use the following directives, which the layout plugin expands at build time:

- `<!-- @sections -->`: a card for each section. The home page uses it.
- `<!-- @cards SECTION -->`: a card for each page in `SECTION`, except the first.
- `<!-- @source PATH -->`: a code block with the file at `PATH`, relative to the
  repository root. To show only some lines, write `PATH#L10-L40`. The Examples and
  Benchmarks pages use it, so the code on the site always matches the repository.
- `<!-- @results NAME -->`: the output of benchmark `NAME` from the results file that
  `RESULTS_FILE` in `layout.js` names.
- `<!-- @reference NAMESPACE -->`: the reference entries for `tl` or `host`, from
  `src/data/reference.js`.

The site has the following files:

- `pages.js`: the sections and their pages, in reading order, with each page's title,
  sidebar group, and description.
- `index.html` and `*/index.html`: the page content.
- `404.html`: the page that Cloudflare Pages serves for unknown URLs.
- `public/_redirects`: the Cloudflare Pages redirects.
- `src/main.js`: highlighting, copy buttons, tabs, and the sidebar's scroll tracking,
  which every page loads.
- `src/explorer.js`, `src/reference.js`, and `src/benchmarks.js`: the tile layout
  explorer, the reference filter, and the benchmark chart, each loaded only by the pages
  that use it.
- `src/data/reference.js`: the reference entries. Update them when a builtin changes
  in `docs/guide/language-reference.md`.
- `src/data/benchmarks.js`: the rows of the overview chart, copied from
  `benchmarks/results/`.
- `src/style.css`: the design tokens for light and dark mode, and the layout.

When you add a benchmark results file, point `RESULTS_FILE` in `layout.js` at it and
update `src/data/benchmarks.js` and the tables on the Benchmarks pages.

The generated MSL in the home page's code panel comes from
`add_kernel.warmup(x, y, out, n, BLOCK=1024).msl`. When code generation changes,
regenerate it and paste the kernel's entry point into `index.html`.
