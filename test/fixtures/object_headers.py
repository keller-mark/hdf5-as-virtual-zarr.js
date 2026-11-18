# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""Generate object_headers_v1.h5 / object_headers_v2.h5: object headers whose
continuation blocks are far (about 1 MB) from the header, because attributes are
added after the data is written. Version 1 object headers (libver="earliest")
and version 2 ("latest")."""
import json
from pathlib import Path

import h5py
import numpy as np

here = Path(__file__).parent
expected = {}
for version, libver in [("v1", "earliest"), ("v2", "latest")]:
    out = here / f"object_headers_{version}.h5"
    with h5py.File(out, "w", libver=libver) as f:
        g = f.create_group("group")
        data = g.create_dataset("data", data=np.arange(131072, dtype="<f8"))  # 1 MiB contiguous
        f.flush()
        # Attributes written after the data do not fit in the original header, so
        # HDF5 adds a continuation block after the data. Few, large attributes keep
        # version 2 headers in compact (not dense) attribute storage.
        for i in range(3):
            data.attrs[f"attr_{i}"] = f"value {i} " + "x" * 1024
            g.attrs[f"group_attr_{i}"] = np.arange(200, dtype="<i4") + i
        f.create_dataset("after", data=np.arange(10, dtype="<i4"))
    with h5py.File(out, "r") as f:
        expected[version] = {
            "data_attrs": {k: v for k, v in f["group/data"].attrs.items()},
            "group_attrs": {k: v.tolist() for k, v in f["group"].attrs.items()},
        }
    print(f"Wrote {out}")
(here / "object_headers.expected.json").write_text(json.dumps(expected, indent=1))
