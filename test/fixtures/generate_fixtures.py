# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anndata>=0.10",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
"""Generate HDF5 (.h5ad) test fixture files using AnnData."""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse


def make_diagonal(dtype: str, m: int, n: int) -> np.ndarray:
    """Create a diagonal-ish matrix for testing."""
    mat = np.zeros((m, n), dtype=dtype)
    for i in range(min(m, n)):
        if i != min(m // 2, n // 2):
            mat[i, i] = i
    return mat


def generate_dense_fixture(output_dir: Path) -> None:
    """Generate an h5ad file with a dense X matrix."""
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


def generate_sparse_fixture(output_dir: Path) -> None:
    """Generate an h5ad file with a sparse CSR X matrix."""
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


def generate_minimal_fixture(output_dir: Path) -> None:
    """Generate a minimal h5ad file."""
    n_obs = 5
    n_var = 3

    X = np.arange(n_obs * n_var, dtype="float32").reshape(n_obs, n_var)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(output_dir / "minimal.h5ad")


def main() -> None:
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_dense_fixture(output_dir)
    generate_sparse_fixture(output_dir)
    generate_minimal_fixture(output_dir)

    print(f"Generated fixtures in {output_dir}")
    for f in sorted(output_dir.glob("*.h5ad")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
