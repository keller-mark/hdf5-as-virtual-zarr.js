# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata>=0.12",
#     "numpy>=1,<3",
#     "pandas>=2,<3",
#     "scipy",
#     "zarr>=3,<4",
# ]
# ///
"""Generate HDF5 (.h5ad) and Zarr v3 (.v3.adata.zarr) test fixture files using AnnData with Zarr v3."""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

anndata.settings.zarr_write_format = 3


def make_diagonal(dtype: str, m: int, n: int) -> np.ndarray:
    """Create a diagonal-ish matrix for testing."""
    mat = np.zeros((m, n), dtype=dtype)
    for i in range(min(m, n)):
        if i != min(m // 2, n // 2):
            mat[i, i] = i
    return mat


def generate_dense_fixture(output_dir: Path) -> None:
    """Generate an h5ad file and zarr v3 store with a dense X matrix."""
    n_obs = 50
    n_var = 25

    X = make_diagonal("float32", n_obs, n_var)

    obs = pd.DataFrame(
        index=[f"obs_{i}" for i in range(n_obs)],
        data={
            "categorical": pd.Categorical(
                [f"cat_{i % 5}" for i in range(n_obs)]
            ),
            "string": np.array([f"str_{i}" for i in range(n_obs)]),
        },
    )
    var = pd.DataFrame(index=[f"var_{i}" for i in range(n_var)])

    obsm = {
        "X_umap": np.random.default_rng(42).standard_normal((n_obs, 2)).astype("float32"),
    }

    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
    )
    adata.write_h5ad(output_dir / "dense.h5ad")
    adata.write_zarr(output_dir / "dense.v3.adata.zarr")


def generate_sparse_fixture(output_dir: Path) -> None:
    """Generate an h5ad file and zarr v3 store with a sparse CSR X matrix."""
    n_obs = 50
    n_var = 25

    X = sparse.csr_matrix(make_diagonal("float32", n_obs, n_var))

    obs = pd.DataFrame(
        index=[f"obs_{i}" for i in range(n_obs)],
        data={
            "categorical": pd.Categorical(
                [f"cat_{i % 5}" for i in range(n_obs)]
            ),
            "string": np.array([f"str_{i}" for i in range(n_obs)]),
        },
    )
    var = pd.DataFrame(index=[f"var_{i}" for i in range(n_var)])

    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=var,
    )
    adata.write_h5ad(output_dir / "sparse.h5ad")
    adata.write_zarr(output_dir / "sparse.v3.adata.zarr")


def generate_minimal_fixture(output_dir: Path) -> None:
    """Generate a minimal h5ad file and zarr v3 store."""
    n_obs = 5
    n_var = 3

    X = np.arange(n_obs * n_var, dtype="float32").reshape(n_obs, n_var)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(output_dir / "minimal.h5ad")
    adata.write_zarr(output_dir / "minimal.v3.adata.zarr")


def main() -> None:
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_dense_fixture(output_dir)
    generate_sparse_fixture(output_dir)
    generate_minimal_fixture(output_dir)

    print(f"Generated v3 fixtures in {output_dir}")
    for f in sorted(output_dir.glob("*.h5ad")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    for f in sorted(output_dir.glob("*.v3.adata.zarr")):
        print(f"  {f.name}/ (zarr v3 directory store)")


if __name__ == "__main__":
    main()
