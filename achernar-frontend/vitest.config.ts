import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
    resolve: {
        // "@/..." -> src/, as in vite.config.ts. Vitest does not load
        // vite.config.ts when this file exists, so the alias is repeated here.
        alias: {
            "@": fileURLToPath(new URL("./src", import.meta.url)),
        },
    },
    test: {
        // Most tests are pure logic or render to static markup, so "node" is
        // the default and they pay nothing for a DOM. A component test that
        // needs one opts in per file with a docblock on its first line:
        //     // @vitest-environment jsdom
        environment: "node",
        include: ["src/**/*.test.{ts,tsx}"],
    },
});
