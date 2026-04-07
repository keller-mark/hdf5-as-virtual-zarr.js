import { parse } from "reference-spec-reader";
import type { AbsolutePath, AsyncReadable } from "@zarrita/storage";

// Copied from @zarrita-storage/src/util.ts
function strip_prefix<Path extends AbsolutePath>(
	path: Path,
): Path extends AbsolutePath<infer Rest> ? Rest : never {
	// @ts-expect-error - TS can't infer this type correctly
	return path.slice(1);
}

// Copied from @zarrita-storage/src/util.ts
function merge_init(
	storeOverrides: RequestInit,
	requestOverrides: RequestInit,
) {
	// Request overrides take precedence over storeOverrides.
	return {
		...storeOverrides,
		...requestOverrides,
		headers: {
			...storeOverrides.headers,
			...requestOverrides.headers,
		},
	};
}

/**
 * Copied from @zarrita-storage/src/util.ts
 * This is for the "binary" loader (custom code is ~2x faster than "atob") from esbuild.
 * https://github.com/evanw/esbuild/blob/150a01844d47127c007c2b1973158d69c560ca21/internal/runtime/runtime.go#L185
 */
let table = new Uint8Array(128);
for (let i = 0; i < 64; i++) {
	table[i < 26 ? i + 65 : i < 52 ? i + 71 : i < 62 ? i - 4 : i * 4 - 205] = i;
}
export function to_binary(base64: string): Uint8Array {
	const n = base64.length;
	const bytes = new Uint8Array(
		// @ts-ignore
		(((n - (base64[n - 1] === "=") - (base64[n - 2] === "=")) * 3) / 4) | 0,
	);
	for (let i = 0, j = 0; i < n; ) {
		const c0 = table[base64.charCodeAt(i++)];
		const c1 = table[base64.charCodeAt(i++)];
		const c2 = table[base64.charCodeAt(i++)];
		const c3 = table[base64.charCodeAt(i++)];
		bytes[j++] = (c0 << 2) | (c1 >> 4);
		bytes[j++] = (c1 << 4) | (c2 >> 2);
		bytes[j++] = (c2 << 6) | c3;
	}
	return bytes;
}

type ReferenceEntry =
	| string
	| [url: string | null]
	| [url: string | null, offset: number, length: number];

// Based on @zarrita-storage/src/ref.ts, but with some differences:
// - `refs` is not a Promise.
// - An internal store is passed in directly, avoiding any assumptions about where the data is coming from.
export class ComposableReferenceStore implements AsyncReadable<RequestInit> {
	#refs: Map<string, ReferenceEntry>;
  #internal_store: AsyncReadable;
	#overrides: RequestInit | undefined;

	constructor(
		refs: Map<string, ReferenceEntry>,
    internal_store: AsyncReadable,
		overrides?: RequestInit,
	) {
		this.#refs = refs;
    this.#internal_store = internal_store;
    this.#overrides = overrides;
	}

	async get(
		key: AbsolutePath,
		opts: RequestInit = {},
	): Promise<Uint8Array | undefined> {
		let ref = this.#refs.get(strip_prefix(key));

		if (!ref) return;

		if (typeof ref === "string") {
			if (ref.startsWith("base64:")) {
				return to_binary(ref.slice("base64:".length));
			}
			return new TextEncoder().encode(ref);
		}

    let [urlOrNull, offset, size] = ref;

    if (offset === undefined || size === undefined || !this.#internal_store.getRange) {
      // TODO: is this correct? should it be .get("/", opts) instead?
      // TODO: throw error in this case?
      return this.#internal_store.get(key, opts);
    }

    const opts_for_get = this.#overrides ? merge_init(this.#overrides, opts) : opts;

		return this.#internal_store.getRange("/", { offset, length: size }, opts_for_get);
  }

  // TODO: does it make sense to support getRange here? get already calls getRange on the internal store...

	static fromSpec(
		spec: Record<string, unknown>,
    internal_store: AsyncReadable,
		overrides?: RequestInit,
	): ComposableReferenceStore {
		// @ts-expect-error - TS doesn't like the type of `parse`
		let refs = parse(spec);
		return new ComposableReferenceStore(refs, internal_store, overrides);
	}
}
