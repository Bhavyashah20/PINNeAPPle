// Browser smoke tests: build the static site, serve it, and drive every mode in Chromium.
import { defineConfig, devices } from "@playwright/test";

const PORT = 8766;
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
  use: { baseURL: `http://localhost:${PORT}/`, trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } }, grepInvert: /@mobile/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, grep: /@mobile/ },
  ],
  webServer: {
    command: `python3 build.py --out dist-test --serve --port ${PORT}`,
    url: `http://localhost:${PORT}/index.html`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
