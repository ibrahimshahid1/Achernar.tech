import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        // Pure logic only: the card builder, the RB-2 criteria and the
        // transport adapter. No DOM environment is needed, and adding one would
        // pull in jsdom for tests that render nothing.
        environment: "node",
        include: ["src/**/*.test.ts"],
    },
});
