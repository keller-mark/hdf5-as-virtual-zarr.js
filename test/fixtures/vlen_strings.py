# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""Generate vlen_strings.h5 (string datasets in every layout) and
vlen_strings.expected.json (values as read by h5py, plus the byte ranges of
the chunks and global heap collections of `many_chunks`, for partial-read tests)."""
import json
import zlib
from pathlib import Path

import h5py
import numpy as np

here = Path(__file__).parent
out = here / "vlen_strings.h5"
utf8 = h5py.string_dtype("utf-8")
ascii_ = h5py.string_dtype("ascii")

SAMPLES = ["", "a", "é", "日本語", "🧬 cell", "naïve T", "x" * 100, "Zürich—Ω"]


def text(i: int) -> str:
    return f"{SAMPLES[i % len(SAMPLES)]}#{i}" if i % 7 else ""


with h5py.File(out, "w") as f:
    # Several gzip chunks, partial last chunk (1003 = 15 * 64 + 43), non-ASCII and empty strings.
    f.create_dataset("chunked_gzip", data=np.array([text(i) for i in range(1003)], dtype=object),
                     dtype=utf8, chunks=(64,), compression="gzip")
    f.create_dataset("chunked_shuffle_gzip", data=np.array([text(i) for i in range(300)], dtype=object),
                     dtype=utf8, chunks=(128,), compression="gzip", shuffle=True)
    f.create_dataset("chunked_2d", data=np.array([[text(r * 11 + c) for c in range(11)] for r in range(9)], dtype=object),
                     dtype=utf8, chunks=(4, 3), compression="gzip")
    f.create_dataset("chunked_uncompressed", data=np.array([text(i) for i in range(50)], dtype=object),
                     dtype=utf8, chunks=(16,))
    # Many chunks: used to check that reading one chunk touches only its own bytes.
    f.create_dataset("many_chunks", data=np.array([f"cell-{i:06d}-{'é' * (i % 5)}" for i in range(20000)], dtype=object),
                     dtype=utf8, chunks=(500,), compression="gzip")
    # Unwritten chunks read as "" (fill value).
    partial = f.create_dataset("chunked_sparse", shape=(100,), chunks=(10,), dtype=utf8)
    partial[25:42] = [text(i) for i in range(17)]
    # Contiguous: more than one 65,536-element virtual chunk.
    f.create_dataset("contiguous_large", data=np.array([f"{i}{'é' if i % 3 else ''}" for i in range(70001)], dtype=object), dtype=utf8)
    f.create_dataset("contiguous_2d", data=np.array([[text(r * 4 + c) for c in range(4)] for r in range(5)], dtype=object), dtype=utf8)
    f.create_dataset("ascii", data=np.array([b"alpha", b"", b"gamma"], dtype=object), dtype=ascii_)
    f.create_dataset("scalar", data="scalar é value", dtype=utf8)
    f.create_dataset("empty", shape=(0,), dtype=utf8)
    # Compact layout (low-level API).
    values = np.array(["compact", "", "ünïcode", "日本"], dtype=object)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    space = h5py.h5s.create_simple((len(values),))
    dsid = h5py.h5d.create(f.id, b"compact", h5py.h5t.py_create(utf8, logical=True), space, dcpl=dcpl)
    h5py.Dataset(dsid)[...] = values
    # LZF is not supported by the JS chunk decoder: reading a chunk must fail clearly.
    f.create_dataset("lzf", data=np.array(["a", "b"], dtype=object), dtype=utf8, chunks=(2,), compression="lzf")
    # Fixed-length byte strings.
    f.create_dataset("fixed_chunked", data=np.array([f"f{i}".encode() for i in range(45)], dtype="S6"),
                     chunks=(10,), compression="gzip")
    f.create_dataset("fixed_contiguous", data=np.array([b"abc", b"", b"abcdef", b"z"], dtype="S6"))

expected: dict = {"values": {}, "shapes": {}}
with h5py.File(out, "r") as f:
    for name, ds in f.items():
        data = ds[()]
        decode = lambda v: v.decode("utf-8") if isinstance(v, bytes) else str(v)
        if ds.shape == ():
            values = [decode(data)]
        else:
            values = [decode(v) for v in np.asarray(data, dtype=object).ravel()]
        expected["values"][name] = values
        expected["shapes"][name] = list(ds.shape)
    assert f["compact"].id.get_create_plist().get_layout() == h5py.h5d.COMPACT
    assert f["contiguous_large"].id.get_create_plist().get_layout() == h5py.h5d.CONTIGUOUS

    ds = f["many_chunks"]
    raw = out.read_bytes()
    chunks = []
    for i in range(ds.id.get_num_chunks()):
        info = ds.id.get_chunk_info(i)
        _, data = ds.id.read_direct_chunk(info.chunk_offset)
        refs = np.frombuffer(zlib.decompress(data), dtype=[("len", "<u4"), ("addr", "<u8"), ("idx", "<u4")])
        chunks.append({
            "index": info.chunk_offset[0] // ds.chunks[0],
            "offset": info.byte_offset,
            "size": info.size,
            "collections": sorted({int(a) for a in refs["addr"] if a}),
        })
    collections = {}
    for c in chunks:
        for addr in c["collections"]:
            assert raw[addr:addr + 4] == b"GCOL"
            collections[addr] = int.from_bytes(raw[addr + 8:addr + 16], "little")
    expected["many_chunks"] = {
        "chunks": chunks,
        "collections": [{"offset": a, "size": s} for a, s in sorted(collections.items())],
    }

(here / "vlen_strings.expected.json").write_text(json.dumps(expected, ensure_ascii=False))
print(f"Wrote {out}")
