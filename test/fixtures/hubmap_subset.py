# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata>=0.12",
#     "numpy>=1,<3",
#     "pandas>=2,<3",
# ]
# ///
#
# Download the `secondary_analysis.h5ad` file for sample `HBM838.LDFP.578` from the HuBMAP Portal at https://portal.hubmapconsortium.org/browse/dataset/5bf1e7b295343c4537206beda25aa4ca
# - https://assets.hubmapconsortium.org/0a21f3fa27109790483f2a0729be53de/secondary_analysis.h5ad

from anndata import read_h5ad


if __name__ == "__main__":
    adata = read_h5ad("secondary_analysis.h5ad")

    small_adata = adata[100, 50].copy()

    small_adata.write_h5ad("secondary_analysis.subset.h5ad")
