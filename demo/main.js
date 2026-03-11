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

/* ---- process file via Web Worker with WORKERFS ---- */
function processFile(file) {
  jsonResult = null;
  downloadBtn.disabled = true;
  output.style.display = "none";
  output.textContent = "";
  fileName = file.name;
  status.textContent = "Converting HDF5 → Zarr reference spec…";

  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    worker.terminate();
    if (e.data.success) {
      jsonResult = JSON.stringify(e.data.refSpec, null, 2);
      output.textContent = jsonResult;
      output.style.display = "block";
      downloadBtn.disabled = false;
      status.textContent = "Done ✓";
    } else {
      status.textContent = "Error: " + e.data.error;
      console.error(e.data.error);
    }
  };
  worker.onerror = (err) => {
    worker.terminate();
    status.textContent = "Error: " + err.message;
    console.error(err);
  };
  worker.postMessage(file);
}
