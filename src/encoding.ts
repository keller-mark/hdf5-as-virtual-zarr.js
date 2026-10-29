/**
 * JSON.stringify with keys sorted alphabetically at all levels.
 */
export function jsonStringifySorted(obj: unknown): string {
  return JSON.stringify(obj, (_key, value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const sorted: Record<string, unknown> = {};
      for (const k of Object.keys(value).sort()) {
        sorted[k] = (value as Record<string, unknown>)[k];
      }
      return sorted;
    }
    return value;
  });
}

/**
 * Inline binary data, matching kerchunk's encoding:
 * - If all bytes are < 128 (ASCII-safe), store as a raw character string
 * - Otherwise, encode as "base64:" + base64
 */
export function inlineBytes(data: Uint8Array): string {
  let binary = "";
  let asciiSafe = true;
  for (let i = 0; i < data.length; i++) {
    if (data[i] >= 128) asciiSafe = false;
    binary += String.fromCharCode(data[i]);
  }
  return asciiSafe ? binary : "base64:" + btoa(binary);
}

/**
 * Decode an inline reference entry to bytes,
 * using the same rules as the reference store.
 */
export function decodeInline(entry: string): Uint8Array {
  if (entry.startsWith("base64:")) {
    const binary = atob(entry.slice("base64:".length));
    return Uint8Array.from(binary, (c) => c.charCodeAt(0));
  }
  return new TextEncoder().encode(entry);
}

/**
 * Encode an array of strings using the vlen-utf8 codec format:
 * uint32 item count, then for each item a uint32 byte length followed by UTF-8 bytes.
 * All integers are little-endian.
 */
export function encodeVlenUtf8(strings: string[]): Uint8Array {
  const encoder = new TextEncoder();
  const encoded = strings.map((s) => encoder.encode(s));
  const totalSize = 4 + encoded.reduce((acc, b) => acc + 4 + b.length, 0);
  const out = new Uint8Array(totalSize);
  const view = new DataView(out.buffer);
  view.setUint32(0, strings.length, true);
  let pos = 4;
  for (const b of encoded) {
    view.setUint32(pos, b.length, true);
    pos += 4;
    out.set(b, pos);
    pos += b.length;
  }
  return out;
}
