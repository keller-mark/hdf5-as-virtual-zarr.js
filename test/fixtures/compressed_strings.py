# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""Generate compressed_strings.h5: chunked, compressed variable-length string datasets."""
from pathlib import Path
import h5py
import numpy as np

out = Path(__file__).parent / "compressed_strings.h5"
str_dtype = h5py.string_dtype("utf-8")
with h5py.File(out, "w") as f:
    # 1-D, gzip, several chunks with a partial final chunk (103 = 10 * 10 + 3).
    f.create_dataset(
        "gzip_1d",
        data=np.array([f"item-{i}-é" for i in range(103)], dtype=object),
        dtype=str_dtype, chunks=(10,), compression="gzip",
    )
    # 2-D, gzip, partial chunks along both axes.
    f.create_dataset(
        "gzip_2d",
        data=np.array([[f"r{r}c{c}" for c in range(5)] for r in range(7)], dtype=object),
        dtype=str_dtype, chunks=(3, 2), compression="gzip",
    )
    # shuffle + gzip.
    f.create_dataset(
        "shuffle_gzip",
        data=np.array([f"s{i}" for i in range(20)], dtype=object),
        dtype=str_dtype, chunks=(8,), compression="gzip", shuffle=True,
    )
print(f"Wrote {out}")
