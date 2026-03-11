import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.test.ts",
  use: {
    baseURL: "http://localhost:5173",
  },
  webServer: {
    command: "pnpm run demo:dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
  },
});
