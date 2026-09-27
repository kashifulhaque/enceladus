import { resolve } from "node:path";

import { defineConfig } from "vite";

import { ALL_PAGES, layout } from "./layout.js";

export default defineConfig({
  // A multi-page app: /docs/quickstart/ serves docs/quickstart/index.html, and unknown URLs 404.
  appType: "mpa",
  plugins: [layout()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(
        ALL_PAGES.map((p) => [p.file.replace(/\/?index\.html$|\.html$/, "") || "index", resolve(import.meta.dirname, p.file)]),
      ),
    },
  },
});
