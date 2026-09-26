import { resolve } from "node:path";

import { defineConfig } from "vite";

import { layout } from "./layout.js";
import { NOT_FOUND, PAGES } from "./pages.js";

export default defineConfig({
  // A multi-page app: /quickstart/ serves quickstart/index.html, and unknown URLs 404.
  appType: "mpa",
  plugins: [layout()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(
        [...PAGES, NOT_FOUND].map((p) => [p.file.replace(/\/?index\.html$|\.html$/, "") || "index", resolve(import.meta.dirname, p.file)]),
      ),
    },
  },
});
