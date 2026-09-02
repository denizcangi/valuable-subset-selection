"""Method 3 — Shapley-value residual importance (diversity only, no queries).

The importance of a set is how far its tuples sit from their own centroid, so a
set of near-identical tuples scores low and a heterogeneous one scores high:

    imp(R') = mean_{t in R'} (1 - sim(t, centroid(R')))

A single tuple's value is then its Shapley value with respect to that set
function — its average marginal contribution over every subset that excludes it:

    resimp(t) = sum_{S subset of R\\{t}} |S|!(|R|-|S|-1)!/|R|! * [imp(S + t) - imp(S)]

R' is the T tuples with the highest residual importance. Note this method never
looks at the query workload; it is purely about how spread out the data is.

The cost is 2^|R| subsets, which caps this at roughly 25 rows in practice. Three
things keep it as fast as it can be: factorials are precomputed once, centroids
are updated incrementally rather than recomputed, and subsets are enumerated
level by level in batches so memory stays bounded.

Run:
    python -m src.method3 --input data/data_iid_uniform_25_5.csv \\
        --threshold 8 --output results/method3.csv
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations, islice
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from .common import (
    get_spark,
    jaccard_distance,
    load_table,
    parse_dataset_name,
    save_selected,
    save_selected_with_scores,
    write_log,
)

SCORE_SCHEMA = StructType(
    [
        StructField("row_id", IntegerType(), False),
        StructField("residual_importance", DoubleType(), False),
    ]
)


class Method3:
    """Diversity-based selection via exact Shapley residual importance."""

    def __init__(self, spark: SparkSession, input_file: str):
        self.spark = spark
        self.input_file = input_file
        self.df = load_table(spark, input_file)
        self.num_rows = self.df.count()
        self.data_cols = [c for c in self.df.columns if c != "row_id"]

    # ------------------------------------------------------------------
    # Set-importance primitives
    # ------------------------------------------------------------------

    @staticmethod
    def compute_centroid(subset_values: List[List[int]]):
        """Centroid of a set of tuples, plus the value counts behind it.

        Values are categorical, so the centroid takes the per-column mode
        rather than a mean; ties break to the smallest value for determinism.
        The counts are returned alongside because they let the caller update
        the centroid in O(D) when one tuple is added, instead of O(|S| * D).
        """
        if not subset_values:
            return [], []

        num_cols = len(subset_values[0])
        counts = [defaultdict(int) for _ in range(num_cols)]
        for row in subset_values:
            for i, value in enumerate(row):
                counts[i][value] += 1

        centroid = []
        for column_counts in counts:
            max_count = max(column_counts.values())
            centroid.append(min(v for v, c in column_counts.items() if c == max_count))

        return centroid, counts

    @staticmethod
    def update_centroid(previous_counts, previous_centroid, new_tuple):
        """Recompute the centroid of S + {t} from the counts of S.

        Only the columns touched by ``new_tuple`` can change their mode, so this
        is O(D) per added tuple. The counts are copied rather than mutated
        because the caller reuses the counts of S across every candidate t.
        """
        if not previous_counts:
            counts = [{value: 1} for value in new_tuple]
            return counts, list(new_tuple)

        counts = [dict(c) for c in previous_counts]
        centroid = list(previous_centroid)

        for column, value in enumerate(new_tuple):
            counts[column][value] = counts[column].get(value, 0) + 1
            max_count = max(counts[column].values())
            centroid[column] = min(v for v, c in counts[column].items() if c == max_count)

        return counts, centroid

    @staticmethod
    def subset_importance(subset_values: List[List[int]], centroid=None) -> float:
        """Average Jaccard distance from each tuple in the set to its centroid."""
        if not subset_values:
            return 0.0
        if len(subset_values) == 1:
            # A single tuple is its own centroid, so distance is zero.
            return 0.0

        if centroid is None:
            centroid, _ = Method3.compute_centroid(subset_values)

        total = sum(jaccard_distance(row, centroid) for row in subset_values)
        return total / float(len(subset_values))

    # ------------------------------------------------------------------
    # Residual importance
    # ------------------------------------------------------------------

    @staticmethod
    def _batched_combinations(ids, k, batch_size):
        """Yield size-k combinations in chunks, so none of it is held at once."""
        iterator = combinations(ids, k)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return
            yield batch

    def compute_residual_importance(
        self, batch_size: int = 10000, slices_factor: int = 4
    ) -> Dict[int, float]:
        """Exact Shapley residual importance for every tuple in R.

        Subsets are processed level by level (all of size 1, then size 2, and so
        on). Within a level, each batch is parallelised across Spark; the worker
        emits ``(tuple_id, contribution)`` pairs which ``reduceByKey`` folds
        together before anything comes back to the driver.
        """
        rows = self.df.select("row_id", *self.data_cols).collect()
        row_map = {r["row_id"]: [r[c] for c in self.data_cols] for r in rows}
        all_ids = list(row_map)
        n = len(all_ids)

        if n <= 1:
            return {row_id: 0.0 for row_id in all_ids}

        total_subsets = 2 ** n
        print(f"Rows: {n}  |  subsets to process: {total_subsets:,}  |  batch size: {batch_size:,}")
        if n > 22:
            print(f"Warning: n={n} means ~{total_subsets:,} subsets. Expect hours.")

        sc = self.spark.sparkContext
        num_cores = sc.defaultParallelism
        bc_row_map = sc.broadcast(row_map)
        # Factorials up to n are computed once here rather than per subset;
        # they are captured by the worker closure and shipped with the task.
        factorials = [math.factorial(i) for i in range(n + 1)]

        def contributions_for_batch(subsets_batch):
            """Marginal contributions of every tuple to every subset in a batch."""
            local_row_map = bc_row_map.value
            local_ids = list(local_row_map)
            out = []

            for subset in subsets_batch:
                S = list(subset)
                if not S:
                    # Empty subsets contribute nothing; skip this one and
                    # keep processing the rest of the batch.
                    continue

                S_set = set(S)
                S_values = [local_row_map[i] for i in S]
                centroid_S, counts_S = Method3.compute_centroid(S_values)
                importance_S = Method3.subset_importance(S_values, centroid=centroid_S)

                # Shapley weight depends only on |S|, so compute it once per
                # subset rather than once per candidate tuple.
                weight = (factorials[len(S)] * factorials[n - len(S) - 1]) / factorials[n]

                for t in local_ids:
                    if t in S_set:
                        continue
                    t_values = local_row_map[t]
                    _, centroid_with_t = Method3.update_centroid(counts_S, centroid_S, t_values)
                    importance_with_t = Method3.subset_importance(
                        S_values + [t_values], centroid=centroid_with_t
                    )
                    out.append((t, weight * (importance_with_t - importance_S)))

            return out

        scores = defaultdict(float)
        processed = 0
        start = time.time()

        for k in range(1, n):
            level_start = time.time()
            for batch in self._batched_combinations(all_ids, k, batch_size):
                num_slices = min(len(batch), max(num_cores * slices_factor, 1))
                rdd = sc.parallelize(batch, numSlices=num_slices)
                results = (
                    rdd.mapPartitions(lambda it: contributions_for_batch(list(it)))
                    .reduceByKey(lambda a, b: a + b)
                    .collect()
                )
                for tuple_id, contribution in results:
                    scores[tuple_id] += contribution
                processed += len(batch)

            elapsed = time.time() - start
            eta = (n - 1 - k) * (elapsed / k)
            print(
                f"  level k={k}/{n-1} done in {time.time() - level_start:.2f}s "
                f"(elapsed {elapsed/60:.1f}min, ETA ~{eta/60:.1f}min)"
            )

        print(f"Processed {processed:,} subsets in {(time.time() - start)/60:.2f} min")
        return dict(scores)

    def select(
        self, threshold: int, batch_size: int = 10000, slices_factor: int = 4
    ) -> Tuple[List[int], DataFrame, float, float]:
        """Rank tuples by residual importance and keep the top T."""
        start = time.time()
        residual = self.compute_residual_importance(
            batch_size=batch_size, slices_factor=slices_factor
        )
        runtime_eval = time.time() - start

        ranked = sorted(residual.items(), key=lambda kv: kv[1], reverse=True)
        selected_ids = [row_id for row_id, _ in ranked[:threshold]]

        if ranked:
            print(f"Highest residual importance: {ranked[0][1]:.8f} (row_id={ranked[0][0]})")
            cutoff = min(threshold - 1, len(ranked) - 1)
            print(f"Lowest selected (rank {cutoff + 1}): {ranked[cutoff][1]:.8f}")

        selected_rows = (
            self.df.filter(F.col("row_id").isin(selected_ids))
            .select("row_id", *self.data_cols)
            .collect()
        )
        selected_values = [[row[c] for c in self.data_cols] for row in selected_rows]
        diversity = self.subset_importance(selected_values)
        print(f"imp(R') = {diversity:.6f}")

        scores_df = self.spark.createDataFrame(
            [(int(rid), float(score)) for rid, score in residual.items()], schema=SCORE_SCHEMA
        )
        return selected_ids, scores_df, diversity, runtime_eval


def run(
    input_file: str,
    threshold: int,
    output_file: str,
    batch_size: int = 10000,
    slices_factor: int = 4,
    save_scores: bool = True,
) -> Tuple[List[int], float]:
    spark = get_spark("method3")
    method = Method3(spark, input_file)

    start = time.time()
    selected_ids, scores_df, diversity, runtime_eval = method.select(
        threshold=threshold, batch_size=batch_size, slices_factor=slices_factor
    )
    runtime_total = time.time() - start

    save_selected(method.df, selected_ids, output_file)
    if save_scores and selected_ids:
        save_selected_with_scores(
            method.df, selected_ids, scores_df, output_file.replace(".csv", "_with_values.csv")
        )

    write_log(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "Method3-ResidualImportance",
            "data_file": parse_dataset_name(input_file),
            "parameters": {"threshold": threshold, "batch_size": batch_size},
            "results": {
                "tuples_selected": len(selected_ids),
                "data_kept_percentage": round(len(selected_ids) / method.num_rows * 100, 4)
                if method.num_rows
                else 0.0,
            },
            "importance": {"diversity_score": float(diversity)},
            "runtimes_sec": {
                "residual_importance": round(runtime_eval, 2),
                "total": round(runtime_total, 2),
            },
        },
        output_file,
    )

    print(f"Total runtime: {runtime_total/60:.2f} min")
    return selected_ids, diversity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method 3: Shapley-based residual importance selection"
    )
    parser.add_argument("--input", required=True, help="CSV file containing the table R")
    parser.add_argument("--threshold", type=int, required=True, help="T, the number of tuples to keep")
    parser.add_argument("--output", required=True, help="CSV file to write R' to")
    parser.add_argument(
        "--batch-size", type=int, default=10000, help="Subsets per Spark batch"
    )
    parser.add_argument(
        "--slices-factor",
        type=int,
        default=4,
        help="Spark partitions per core within a batch",
    )
    parser.add_argument("--no-scores", action="store_true", help="Skip the *_with_values.csv file")
    args = parser.parse_args()

    run(
        args.input,
        args.threshold,
        args.output,
        batch_size=args.batch_size,
        slices_factor=args.slices_factor,
        save_scores=not args.no_scores,
    )


if __name__ == "__main__":
    main()
