"""Query workload generator.

Methods 1 and 2 need a set of queries to define tuple popularity. Sampling
predicates uniformly at random would give almost every tuple a popularity of 0
or 1, which makes popularity useless as a ranking signal. So the generator
deliberately builds in the two properties real workloads have:

  Skew    Predicate values are sampled from each column's own empirical
          distribution, biased further towards frequent values. Queries ask
          about values that actually exist and that people actually search for.
  Overlap A pool of "popular patterns" (attribute-value combinations) is built
          up front and reused across queries, so the same tuples get returned
          repeatedly and popularity spreads over a realistic range.

Queries are conjunctions of equality predicates, stored one per row as JSON.
Output columns: query_id, k (predicate count), from_pattern, selectivity
(number of matching tuples), predicates_json.

Run:
    python -m src.query_generator --data data/data_iid_uniform_100_5.csv \\
        --num-queries 1000 --outdir queries
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class QueryGenerator:
    """Generates conjunctive equality queries with controlled skew and overlap."""

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df = df
        self.rng = np.random.default_rng(seed)

        self.num_rows = len(df)
        self.num_cols = len(df.columns)
        self.columns = df.columns.tolist()

        self.empirical_distributions: Dict[str, dict] = {}
        self.popular_values: Dict[str, list] = {}
        self.inverted_index: Dict[str, Dict[int, np.ndarray]] = {}

        self._analyse_distributions()
        self._build_inverted_index()

    def _analyse_distributions(self) -> None:
        """Record each column's value distribution and its most frequent values."""
        for col in self.columns:
            counts = self.df[col].value_counts()
            self.empirical_distributions[col] = {
                "values": counts.index.tolist(),
                "counts": counts.values.tolist(),
                "probabilities": (counts / len(self.df)).values.tolist(),
            }
            head_size = max(1, len(counts) // 5)  # top 20% of distinct values
            self.popular_values[col] = counts.head(head_size).index.tolist()

    def _build_inverted_index(self) -> None:
        """Map (column, value) -> row indices, so queries evaluate by set intersection."""
        for col in self.columns:
            groups = self.df.groupby(col).groups
            self.inverted_index[col] = {int(v): idx.values for v, idx in groups.items()}

    def _sample_value(
        self,
        col: str,
        bias_towards_popular: bool = True,
        bias_exponent: float = 1.5,
        head_weight: float = 0.2,
    ) -> int:
        """Sample a value that exists in ``col``, optionally biased to frequent ones.

        Two knobs, applied in order: raising the empirical probabilities to
        ``bias_exponent`` and renormalising sharpens the existing skew, then
        ``head_weight`` mixes in a uniform distribution over just the popular
        values, creating explicit hotspots even in a flat column.
        """
        distribution = self.empirical_distributions[col]
        probabilities = np.array(distribution["probabilities"])

        if bias_towards_popular:
            probabilities = np.power(probabilities, bias_exponent)
            probabilities /= probabilities.sum()

            if head_weight > 0:
                is_popular = np.isin(distribution["values"], self.popular_values[col])
                head = is_popular / max(is_popular.sum(), 1)
                probabilities = (1 - head_weight) * probabilities + head_weight * head
                probabilities /= probabilities.sum()

        index = self.rng.choice(len(distribution["values"]), p=probabilities)
        return distribution["values"][index]

    def execute_query(self, query: Dict) -> np.ndarray:
        """Row indices satisfying every predicate, via inverted-index intersection."""
        matches = []
        for attribute, value in query["predicates"].items():
            rows = self.inverted_index[attribute].get(int(value))
            if rows is None:
                return np.array([], dtype=int)
            matches.append(rows)

        if not matches:
            return np.arange(self.num_rows, dtype=int)

        return reduce(lambda a, b: np.intersect1d(a, b, assume_unique=False), matches)

    def selectivity(self, query: Dict) -> int:
        return int(self.execute_query(query).size)

    def _generate_until_nonempty(self, generator_fn, retries: int = 100) -> Dict:
        """Retry generation until the query returns at least one tuple.

        Empty queries carry no popularity information, and long conjunctions on
        a sparse table produce them often, so it's cheaper to resample than to
        keep them and filter later.
        """
        query = generator_fn()
        for _ in range(retries):
            if self.selectivity(query) > 0:
                return query
            query = generator_fn()
        return query

    def _create_popular_patterns(
        self, num_patterns: int, max_size: int, bias_exponent: float, head_weight: float
    ) -> List[Dict]:
        """Build reusable attribute-value combinations to share across queries.

        Patterns start at size 2: a single-attribute pattern is not really a
        shared "shape", it's just a popular value, which the sampler already
        produces on its own.
        """
        patterns = []
        for _ in range(num_patterns):
            size = min(int(self.rng.integers(2, max_size + 1)), self.num_cols)
            attributes = self.rng.choice(self.columns, size=size, replace=False).tolist()
            patterns.append(
                {
                    attribute: self._sample_value(
                        attribute,
                        bias_towards_popular=True,
                        bias_exponent=bias_exponent,
                        head_weight=head_weight,
                    )
                    for attribute in attributes
                }
            )
        return patterns

    def _query_from_pattern(
        self, patterns: List[Dict], max_predicates: int
    ) -> Dict:
        """Reuse a popular pattern, optionally extending it with extra predicates."""
        predicates = dict(self.rng.choice(patterns))
        target_size = int(self.rng.integers(len(predicates), max_predicates + 1))

        while len(predicates) < target_size:
            available = [c for c in self.columns if c not in predicates]
            if not available:
                break
            column = self.rng.choice(available)
            predicates[column] = self._sample_value(
                column, bias_towards_popular=True, bias_exponent=1.5
            )

        return {"predicates": predicates, "from_pattern": True}

    def _fresh_query(
        self,
        min_predicates: int,
        max_predicates: int,
        use_popular: bool,
        bias_exponent: float,
        head_weight: float,
    ) -> Dict:
        """Generate a query from scratch, without reusing any pattern."""
        k = int(self.rng.integers(min_predicates, max_predicates + 1))
        columns = self.rng.choice(self.columns, size=k, replace=False).tolist()
        predicates = {
            column: self._sample_value(
                column,
                bias_towards_popular=use_popular,
                bias_exponent=bias_exponent,
                head_weight=head_weight,
            )
            for column in columns
        }
        return {"predicates": predicates, "from_pattern": False}

    def generate(
        self,
        num_queries: int,
        min_predicates: int = 1,
        max_predicates: Optional[int] = None,
        popular_query_ratio: float = 0.7,
        overlap_patterns: int = 10,
        bias_exponent: float = 1.5,
        head_weight: float = 0.2,
        require_nonempty: bool = True,
        allow_duplicates: bool = True,
    ) -> List[Dict]:
        """Generate the workload.

        ``popular_query_ratio`` of queries bias towards frequent values, and
        half of those additionally reuse a stored pattern — so roughly 35% of
        the workload shares structure with another query, which is what creates
        a meaningful spread of tuple popularities.
        """
        if max_predicates is None:
            max_predicates = min(5, self.num_cols)

        patterns = self._create_popular_patterns(
            overlap_patterns, max_predicates, bias_exponent, head_weight
        )

        queries: List[Dict] = []
        seen = set()
        attempts = 0
        max_attempts = num_queries * 3

        while len(queries) < num_queries and attempts < max_attempts:
            attempts += 1
            use_popular = self.rng.random() < popular_query_ratio
            use_pattern = use_popular and patterns and self.rng.random() < 0.5

            if use_pattern:
                query = self._generate_until_nonempty(
                    lambda: self._query_from_pattern(patterns, max_predicates)
                )
            else:
                query = self._generate_until_nonempty(
                    lambda: self._fresh_query(
                        min_predicates, max_predicates, use_popular, bias_exponent, head_weight
                    )
                )

            if not allow_duplicates:
                key = tuple(sorted(query["predicates"].items()))
                if key in seen:
                    continue
                seen.add(key)

            query["query_id"] = len(queries)
            query["k"] = len(query["predicates"])
            query["selectivity"] = self.selectivity(query)
            query["from_pattern"] = bool(query.get("from_pattern", False))
            queries.append(query)

        empty = sum(1 for q in queries if q["selectivity"] == 0)
        if empty:
            print(f"{empty}/{len(queries)} generated queries returned no tuples")
        if require_nonempty:
            queries = [q for q in queries if q["selectivity"] > 0]
            print(f"Kept {len(queries)} non-empty queries")

        return queries


def parse_data_filename(path: str):
    """Extract distribution / rows / cols from a generated dataset filename."""
    name = os.path.basename(path)
    match = re.match(r"^data_([A-Za-z0-9_]+)_(\d+)([kK])?_(\d+)\.csv$", name)
    if not match:
        return None, None, None
    distribution, rows, k_flag, cols = match.groups()
    return distribution, int(rows) * (1000 if k_flag else 1), int(cols)


def generate_for_dataset(
    data_file: str,
    num_queries: int,
    outdir: str,
    max_predicates: Optional[int] = None,
    seed: int = 42,
) -> str:
    df = pd.read_csv(data_file)
    distribution, num_rows, num_cols = parse_data_filename(data_file)
    distribution = distribution or "unknown"
    num_rows = num_rows or len(df)
    num_cols = num_cols or len(df.columns)

    if max_predicates is None:
        # Half the columns: long conjunctions on a wide table almost never
        # match anything, which wastes generation attempts.
        max_predicates = max(1, num_cols // 2)

    generator = QueryGenerator(df, seed=seed)
    queries = generator.generate(num_queries=num_queries, max_predicates=max_predicates)

    rows = [
        {
            "query_id": q["query_id"],
            "k": q["k"],
            "from_pattern": q["from_pattern"],
            "selectivity": q["selectivity"],
            "predicates_json": json.dumps(
                {key: int(q["predicates"][key]) for key in sorted(q["predicates"])}
            ),
        }
        for q in queries
    ]

    os.makedirs(outdir, exist_ok=True)
    rows_label = f"{num_rows // 1000}k" if num_rows >= 1000 and num_rows % 1000 == 0 else str(num_rows)
    path = os.path.join(
        outdir, f"query_{num_queries}_{distribution}_{rows_label}_{num_cols}.csv"
    )
    pd.DataFrame(rows).to_csv(
        path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar=None
    )
    print(f"Wrote {len(rows)} queries to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a query workload for a dataset")
    parser.add_argument("--data", required=True, nargs="+", help="Dataset CSV file(s)")
    parser.add_argument("--num-queries", type=int, default=1000, help="Queries per dataset")
    parser.add_argument("--outdir", default="queries", help="Directory to write query CSVs into")
    parser.add_argument(
        "--max-predicates",
        type=int,
        default=None,
        help="Maximum predicates per query (default: half the column count)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    for data_file in args.data:
        generate_for_dataset(
            data_file,
            args.num_queries,
            args.outdir,
            max_predicates=args.max_predicates,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
