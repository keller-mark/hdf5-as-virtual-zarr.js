import { test, expect } from "@playwright/test";
import * as h5wasm from "h5wasm/node";
import { join } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

const tmpDir = mkdtempSync(join(tmpdir(), "h5-demo-test-"));
const h5Path = join(tmpDir, "test.h5");

test.beforeAll(async () => {
  await h5wasm.ready;
  const f = new h5wasm.File(h5Path, "w");
  f.create_dataset({
    name: "data",
    data: new Float32Array([1, 2, 3, 4, 5, 6]),
    shape: [2, 3],
    dtype: "<f",
  });
  f.close();
});

test("converts HDF5 file and displays JSON reference spec", async ({ page }) => {
  await page.goto("/");

  const fileInput = page.locator("#file-input");
  await fileInput.setInputFiles(h5Path);

  const output = page.locator("#output");
  await expect(output).toBeVisible({ timeout: 30_000 });

  const text = await output.textContent();
  const json = JSON.parse(text);
  expect(json).toHaveProperty("version", 1);
  expect(json).toHaveProperty("refs");
  expect(json.refs).toHaveProperty([".zgroup"]);

  const downloadBtn = page.locator("#download-btn");
  await expect(downloadBtn).toBeEnabled();
});
