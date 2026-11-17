/**
 * Decoding of HDF5 filter pipelines for chunks that must be read during translation
 * (e.g. chunked variable-length string datasets, whose chunks hold global heap references).
 *
 * Uses only web-standard APIs (DecompressionStream) so it works in browsers and Node.
 */

const FILTER_DEFLATE = 1;
const FILTER_SHUFFLE = 2;
const FILTER_FLETCHER32 = 3;

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  // HDF5's deflate filter writes zlib streams, i.e. the "deflate" format.
  const stream = new Blob([data as Uint8Array<ArrayBuffer>])
    .stream()
    .pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function unshuffle(data: Uint8Array, elementSize: number): Uint8Array {
  if (elementSize <= 1) return data;
  const count = Math.floor(data.length / elementSize);
  const out = new Uint8Array(data.length);
  for (let b = 0; b < elementSize; b++) {
    const src = b * count;
    for (let i = 0; i < count; i++) {
      out[i * elementSize + b] = data[src + i];
    }
  }
  // Trailing bytes (if any) are left unshuffled by HDF5.
  out.set(data.subarray(count * elementSize), count * elementSize);
  return out;
}

/**
 * Reverse an HDF5 filter pipeline for a single chunk.
 * @param data Raw (filtered) chunk bytes, as stored in the file.
 * @param pipeline Parsed filter pipeline message (in the order filters were applied on write).
 * @param filterMask Per-chunk filter mask; bit i set means filter i was skipped for this chunk.
 */
export async function decodeChunk(
  data: Uint8Array,
  pipeline: Map<string, unknown>[] | null,
  filterMask = 0,
): Promise<Uint8Array> {
  if (!pipeline) return data;
  let out = data;
  for (let i = pipeline.length - 1; i >= 0; i--) {
    if (filterMask & (1 << i)) continue;
    const filter = pipeline[i];
    const id = filter.get("filter_id") as number;
    const clientData = (filter.get("client_data") as number[] | undefined) ?? [];
    if (id === FILTER_DEFLATE) {
      out = await inflate(out);
    } else if (id === FILTER_SHUFFLE) {
      out = unshuffle(out, clientData[0] ?? 1);
    } else if (id === FILTER_FLETCHER32) {
      out = out.subarray(0, out.length - 4);
    } else {
      throw new Error(`Unsupported HDF5 filter id ${id} for chunk decoding`);
    }
  }
  return out;
}
