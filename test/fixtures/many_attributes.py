# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""Generate many_attributes_v1.h5 / many_attributes_v2.h5: groups and datasets with
many attributes added after creation, so that object headers span several
continuation blocks (version 1 headers, libver="earliest") or attributes move to
dense storage in a fractal heap (version 2 headers, libver="latest").
Writes many_attributes.expected.json with the attributes as read by h5py."""
import json
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

here = Path(__file__).parent


def plain(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()] if v.dtype.kind in "OS" else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


expected = {}
for version, libver in [("v1", "earliest"), ("v2", "latest")]:
    out = here / f"many_attributes_{version}.h5"
    with h5py.File(out, "w", libver=libver) as f:
        g = f.create_group("group")
        data = g.create_dataset("data", data=np.arange(1000, dtype="<f8"))
        f.flush()
        for i in range(40):
            data.attrs[f"attr_{i:02d}"] = f"value {i} " + "x" * (i * 3)
            g.attrs[f"group_attr_{i:02d}"] = i
        # anndata-style attributes, added last.
        data.attrs["encoding-type"] = "array"
        data.attrs["encoding-version"] = "0.2.0"
        g.attrs["encoding-type"] = "dict"
        g.attrs["encoding-version"] = "0.1.0"
        # Mixed types.
        g.attrs["floats"] = np.linspace(0, 1, 5)
        g.attrs["strings"] = np.array(["a", "bé", ""], dtype=object)
    with h5py.File(out, "r") as f:
        expected[version] = {
            path: {k: plain(v) for k, v in f[path].attrs.items()} for path in ["group", "group/data"]
        }
    print(f"Wrote {out}")

(here / "many_attributes.expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False))

if "--dump" in sys.argv:
    # Print the version 1 object header blocks of group/data.
    raw = (here / "many_attributes_v1.h5").read_bytes()
    with h5py.File(here / "many_attributes_v1.h5", "r") as f:
        addr = h5py.h5o.get_info(f["group/data"].id).addr
    version, _, nmsgs, _, size = struct.unpack_from("<BBHII", raw, addr)
    print(f"header at {addr}: version {version}, {nmsgs} messages, first block size {size}")
    blocks = [(addr + 16, size)]
    while blocks:
        start, size = blocks.pop(0)
        print(f"block at {start}, size {size}")
        pos = start
        while pos + 8 <= start + size:
            mtype, msize, flags = struct.unpack_from("<HHB", raw, pos)
            print(f"  +{pos - start:5d} type {mtype:3d} size {msize:5d} flags {flags}")
            if mtype == 0x10:
                blocks.append(struct.unpack_from("<QQ", raw, pos + 8))
            pos += 8 + msize
