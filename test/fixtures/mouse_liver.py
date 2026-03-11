# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anndata>=0.11",
#     "kerchunk",
#     "numpy>=1,<3",
#     "pandas>=2,<3",
#     "scipy",
#     "zarr>=2,<3",
# ]
# ///
"""Generate HDF5 (.h5ad), Zarr (.adata.zarr), kerchunk reference spec (.h5ad.refspec.json), and Zarr consolidated metadata (.adata.zmetadata.json) test fixture files using AnnData."""

import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import zarr
from kerchunk.hdf import SingleHdf5ToZarr
from scipy import sparse
from os.path import join

def generate_h5ad_ref_spec(h5_url, omit_url=True):
    h5chunks = SingleHdf5ToZarr(h5_url, inline_threshold=300)
    h5dict = h5chunks.translate()
    if omit_url:
        for key, val in h5dict['refs'].items():
            if isinstance(val, list):
                h5dict['refs'][key] = [None, *val[1:]]
    return h5dict

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = join(str(output_dir), "mouse_liver.h5ad")

    output_dict = generate_h5ad_ref_spec(input_path)
    refspec_path = output_dir / "mouse_liver.h5ad.refspec.json"

    with open(refspec_path, "w") as f:
        json.dump(output_dict, f)
