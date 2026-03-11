import { test, expect } from "@playwright/test";
import { join } from "path";

const h5Path = join(import.meta.dirname, "../test/fixtures/minimal.h5ad");

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
