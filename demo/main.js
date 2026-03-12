import { SingleHdf5ToZarr } from "hdf5-as-virtual-zarr";

/**
 * AsyncReadable implementation that reads from a browser File using slice().
 * Only the requested byte ranges are read — no full-file load needed.
 */
class FileSource {
  constructor(file) {
    this.file = file;
    this._getRangeCalls = [];
  }

  async get() {
    return new Uint8Array(await this.file.arrayBuffer());
  }

  async getRange(_key, range) {
    let offset, length;
    if ("suffixLength" in range) {
      offset = this.file.size - range.suffixLength;
      length = range.suffixLength;
    } else {
      offset = range.offset;
      length = range.length;
    }
    const blob = this.file.slice(offset, offset + length);
    const ab = await blob.arrayBuffer();
    this._getRangeCalls.push({ offset, length: ab.byteLength });
    return new Uint8Array(ab);
  }

  uniqueBytesFetched() {
    if (this._getRangeCalls.length === 0) return 0;
    const intervals = this._getRangeCalls
      .map(({ offset, length }) => [offset, offset + length])
      .sort((a, b) => a[0] - b[0]);
    let total = 0;
    let [curStart, curEnd] = intervals[0];
    for (let i = 1; i < intervals.length; i++) {
      const [start, end] = intervals[i];
      if (start <= curEnd) {
        curEnd = Math.max(curEnd, end);
      } else {
        total += curEnd - curStart;
        curStart = start;
        curEnd = end;
      }
    }
    return total + (curEnd - curStart);
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

    const bytesRead = source.uniqueBytesFetched();
    const pct = ((bytesRead / file.size) * 100).toFixed(1);
    const fmt = (n) => n >= 1024 * 1024
      ? (n / (1024 * 1024)).toFixed(2) + " MB"
      : (n / 1024).toFixed(1) + " KB";
    status.textContent = `Done ✓ — read ${fmt(bytesRead)} of ${fmt(file.size)} (${pct}%)`;
  } catch (err) {
    status.textContent = "Error: " + err.message;
    console.error(err);
  }
}
