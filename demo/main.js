import { SingleHdf5ToZarr } from "hdf5-to-zarr";

/**
 * Source implementation that reads from a browser File using slice().
 * Only the requested byte ranges are read — no full-file load needed.
 */
class FileSource {
  type = "file";

  constructor(file) {
    this.file = file;
    this.url = new URL("file:///" + encodeURIComponent(file.name));
    this.metadata = { size: file.size };
  }

  async fetch(offset, length) {
    const blob = length !== undefined
      ? this.file.slice(offset, offset + length)
      : this.file.slice(offset);
    return blob.arrayBuffer();
  }
}

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const downloadBtn = document.getElementById("download-btn");
const output = document.getElementById("output");
const status = document.getElementById("status");

let jsonResult = null;
let fileName = "";

/* ---- drag & drop ---- */
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});

/* ---- file input ---- */
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) processFile(file);
});

/* ---- download ---- */
downloadBtn.addEventListener("click", () => {
  if (!jsonResult) return;
  const blob = new Blob([jsonResult], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName.replace(/\.[^.]+$/, "") + ".refspec.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 100);
});

/* ---- process file using partial reads (no worker needed) ---- */
async function processFile(file) {
  jsonResult = null;
  downloadBtn.disabled = true;
  output.style.display = "none";
  output.textContent = "";
  fileName = file.name;
  status.textContent = "Converting HDF5 → Zarr reference spec…";

  try {
    const source = new FileSource(file);
    const converter = new SingleHdf5ToZarr(source, { url: file.name });
    const refSpec = await converter.translate();

    jsonResult = JSON.stringify(refSpec, null, 2);
    output.textContent = jsonResult;
    output.style.display = "block";
    downloadBtn.disabled = false;
    status.textContent = "Done ✓";
  } catch (err) {
    status.textContent = "Error: " + err.message;
    console.error(err);
  }
}
