"""Synthetic dataset generator.

There is no given dataset for this problem, so we generate tables whose
statistical shape stresses the three methods differently:

  iid_uniform     Every value equally likely. Few duplicate values, so tuples
                  are mutually distant and queries match few rows. The baseline.
  popular_values  Some columns are Zipf-skewed, so a handful of values dominate.
                  Many tuples share values, which inflates popularity and makes
                  tuples similar — the hard case for both LSH and diversity.
  clustered       Rows are drawn around a few per-column centres with noise, so
                  the table has dense groups. This is where diversity-aware
                  selection should visibly beat pure popularity, since picking
                  one representative per cluster maximises spread.

Output files are named ``data_<distribution>_<rows>_<cols>.csv`` (or
``..._<rows>k_...`` for exact thousands), which the methods parse for run logs.

Run:
    python -m src.data_generator --rows 100 --cols 5 --outdir data
    python -m src.data_generator --rows 50000 --cols 10 --distributions clustered
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


class DatasetGenerator:
    """Generates integer tables under three configurable distributions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_iid_uniform(
        self, num_rows: int, num_cols: int, domain_size: int = 100
    ) -> pd.DataFrame:
        """Each column drawn independently and uniformly over [0, domain_size)."""
        columns = [f"A{i}" for i in range(num_cols)]
        data = np.empty((num_rows, num_cols), dtype=np.int32)
        for j in range(num_cols):
            data[:, j] = self.rng.integers(0, domain_size, size=num_rows)
        return pd.DataFrame(data, columns=columns)

    def _zipf_values(self, domain_size: int, alpha: float, size: int) -> np.ndarray:
        """Sample from a Zipf-like distribution over [0, domain_size).

        Probability of rank k is proportional to 1/k^alpha, normalised over the
        finite domain. Larger alpha concentrates more mass on the first values.
        """
        ranks = np.arange(1, domain_size + 1)
        probabilities = 1.0 / np.power(ranks, alpha)
        probabilities /= probabilities.sum()
        return self.rng.choice(domain_size, size=size, p=probabilities)

    def generate_popular_values(
        self,
        num_rows: int,
        num_cols: int,
        domain_size: int = 100,
        zipf_param: float = 1.5,
        skewed_col_ratio: float = 0.7,
    ) -> pd.DataFrame:
        """Skew a fraction of the columns so a few values dominate them.

        Leaving the remaining columns uniform is deliberate: real tables mix
        heavily-repeated fields (status, country) with near-unique ones, and
        that mix is what makes popularity and similarity pull apart.
        """
        columns = [f"A{i}" for i in range(num_cols)]
        data = np.empty((num_rows, num_cols), dtype=np.int32)

        num_skewed = int(num_cols * skewed_col_ratio)
        skewed = set(self.rng.choice(num_cols, size=num_skewed, replace=False))

        for j in range(num_cols):
            if j in skewed:
                data[:, j] = self._zipf_values(domain_size, zipf_param, num_rows)
            else:
                data[:, j] = self.rng.integers(0, domain_size, size=num_rows)

        return pd.DataFrame(data, columns=columns)

    def generate_clustered(
        self,
        num_rows: int,
        num_cols: int,
        cluster_count: int = 4,
        domain_size: int = 100,
        within_cluster_variance: float = 0.15,
    ) -> pd.DataFrame:
        """Draw rows around per-cluster centres with Gaussian noise.

        Cluster sizes decay geometrically so the groups are uneven, as real
        clusters are. ``within_cluster_variance`` is a fraction of the domain:
        lower values make clusters tighter and the table more redundant.
        """
        columns = [f"A{i}" for i in range(num_cols)]

        decay = 0.8
        raw = np.array([decay ** i for i in range(cluster_count)])
        cluster_probs = raw / raw.sum()
        assignments = self.rng.choice(cluster_count, size=num_rows, p=cluster_probs)

        centres = np.zeros((num_cols, cluster_count), dtype=int)
        for j in range(num_cols):
            centres[j, :] = self.rng.integers(0, domain_size, size=cluster_count)

        data = np.empty((num_rows, num_cols), dtype=np.int32)
        max_noise = within_cluster_variance * domain_size

        for cluster_id in range(cluster_count):
            mask = assignments == cluster_id
            size = int(mask.sum())
            if size == 0:
                continue
            for j in range(num_cols):
                noise = self.rng.normal(0, max_noise, size=size)
                values = np.clip(np.round(centres[j, cluster_id] + noise), 0, domain_size - 1)
                data[mask, j] = values.astype(int)

        return pd.DataFrame(data, columns=columns)


def rows_token(num_rows: int) -> str:
    """Format the row count for filenames: 50000 -> '50k', 250 -> '250'."""
    if num_rows >= 1000 and num_rows % 1000 == 0:
        return f"{num_rows // 1000}k"
    return str(num_rows)


GENERATORS = {
    "iid_uniform": "generate_iid_uniform",
    "popular_values": "generate_popular_values",
    "clustered": "generate_clustered",
}


def generate(
    distribution: str,
    num_rows: int,
    num_cols: int,
    outdir: str,
    domain_size: int = 100,
    seed: int = 42,
) -> str:
    generator = DatasetGenerator(seed=seed)
    method = getattr(generator, GENERATORS[distribution])
    df = method(num_rows=num_rows, num_cols=num_cols, domain_size=domain_size)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(
        outdir, f"data_{distribution}_{rows_token(num_rows)}_{num_cols}.csv"
    )
    df.to_csv(path, index=False)
    print(f"Wrote {num_rows} x {num_cols} {distribution} table to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic integer tables")
    parser.add_argument("--rows", type=int, required=True, help="Number of tuples")
    parser.add_argument("--cols", type=int, required=True, help="Number of attributes")
    parser.add_argument("--outdir", default="data", help="Directory to write CSVs into")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=list(GENERATORS),
        choices=list(GENERATORS),
        help="Which distributions to generate (default: all three)",
    )
    parser.add_argument(
        "--domain-size", type=int, default=100, help="Values per column: [0, domain_size)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    for distribution in args.distributions:
        generate(
            distribution,
            args.rows,
            args.cols,
            args.outdir,
            domain_size=args.domain_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
