"""Method 1 — popularity-based selection.

The importance of a tuple is the number of queries whose answer set contains
it, and the importance of a subset is the share of total popularity it covers:

    imp(t)  = pop(t)
    imp(R') = sum_{t in R'} pop(t) / sum_{t in R} pop(t)

Because tuple importances are independent of each other, the optimal R' is
simply the T most popular tuples — no search over subsets is needed. That makes
this the only method here that scales to hundreds of thousands of rows.

Run:
    python -m src.method1 --input data/data_iid_uniform_50k_10.csv \\
        --queries queries/query_1000_iid_uniform_50k_10.csv \\
        --threshold 100 --output results/method1.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, MapType, StringType

from .common import (
    get_spark,
    load_table,
    parse_dataset_name,
    save_selected,
    save_selected_with_scores,
    write_log,
)


class Method1:
    """Popularity-based selection of the top-T tuples."""

    def __init__(self, spark: SparkSession, input_file: str):
        self.spark = spark
        self.input_file = input_file
        self.df = load_table(spark, input_file)
        self.columns = self.df.columns
        self.data_cols = [c for c in self.columns if c != "row_id"]
        self.num_rows = self.df.count()

    def compute_popularity(self, query_file_path: str) -> Tuple[DataFrame, int]:
        """Count, for each tuple, how many queries return it.

        The query file stores each query's predicates as a JSON object in
        ``predicates_json`` plus ``k``, the number of predicates. Rather than
        running |Q| separate filters, we evaluate the whole workload as a single
        distributed join:

        1. Explode each query into one row per predicate -> (query_id, k, col, value).
        2. Represent each tuple as a map from column name to value.
        3. Join on ``tuple_map[col] == value`` — this matches any tuple
           satisfying *at least one* predicate of the query.
        4. Count matched predicates per (query_id, row_id) and keep only the
           pairs whose count equals k, i.e. all predicates satisfied.

        Returns the popularity DataFrame (tuples with pop > 0 only) and the
        total popularity over R, which is the denominator of imp(R').
        """
        queries_pdf = pd.read_csv(query_file_path)
        queries_df = self.spark.createDataFrame(queries_pdf)

        predicate_schema = MapType(StringType(), IntegerType())
        predicates_df = queries_df.select(
            "query_id",
            "k",
            F.explode(F.from_json("predicates_json", predicate_schema)).alias("col", "value"),
        )

        tuple_map = F.map_from_arrays(
            F.array([F.lit(c) for c in self.data_cols]),
            F.array([F.col(c) for c in self.data_cols]),
        )
        tuples_df = self.df.select("row_id", tuple_map.alias("col_value_pairs"))

        partial_matches = predicates_df.join(
            tuples_df,
            tuples_df.col_value_pairs[F.col("col")] == F.col("value"),
            "inner",
        ).select("query_id", "k", "row_id")

        match_counts = partial_matches.groupBy("query_id", "row_id").agg(
            F.count("*").alias("match_count")
        )
        full_matches = match_counts.join(
            queries_df.select("query_id", "k"), "query_id", "inner"
        ).filter(F.col("match_count") == F.col("k"))

        popularity_df = (
            full_matches.groupBy("row_id")
            .agg(F.count("query_id").alias("popularity"))
            .orderBy("row_id")
        )
        total_popularity = full_matches.select("query_id", "row_id").count()

        if total_popularity == 0:
            empty = self.spark.createDataFrame([], "row_id INT, popularity INT")
            return empty, 0

        return popularity_df.cache(), total_popularity

    def select(self, threshold: int, query_file_path: str):
        """Return the T most popular tuples and the importance of that set."""
        popularity_df, total_popularity = self.compute_popularity(query_file_path)

        top = (
            popularity_df.orderBy(F.desc("popularity"), F.asc("row_id"))
            .limit(threshold)
            .collect()
        )
        selected_ids: List[int] = [row.row_id for row in top]
        selected_popularity = sum(row.popularity for row in top)
        importance = selected_popularity / total_popularity if total_popularity else 0.0

        return selected_ids, importance, popularity_df, total_popularity


def run(
    input_file: str,
    query_file: str,
    threshold: int,
    output_file: str,
    save_scores: bool = True,
) -> Tuple[List[int], float]:
    spark = get_spark("method1")
    method = Method1(spark, input_file)

    start = time.time()
    selected_ids, importance, popularity_df, total_popularity = method.select(
        threshold=threshold, query_file_path=query_file
    )
    runtime = time.time() - start

    print(f"Selected {len(selected_ids)} tuples, imp(R') = {importance:.6f}")
    print(f"Selection took {runtime:.2f}s")

    save_selected(method.df, selected_ids, output_file)
    if save_scores:
        scored = output_file.replace(".csv", "_with_values.csv")
        save_selected_with_scores(method.df, selected_ids, popularity_df, scored)

    queries_pdf = pd.read_csv(query_file)
    write_log(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "Method1-Popularity",
            "data_file": parse_dataset_name(input_file),
            "query_file": {
                "filename": query_file.split("/")[-1],
                "num_queries": len(queries_pdf),
                "avg_k": float(queries_pdf["k"].mean()) if "k" in queries_pdf else None,
                "avg_selectivity": (
                    float(queries_pdf["selectivity"].mean())
                    if "selectivity" in queries_pdf
                    else None
                ),
            },
            "threshold": threshold,
            "results": {
                "tuples_selected": len(selected_ids),
                "data_kept_percentage": round(len(selected_ids) / method.num_rows * 100, 4),
                "tuples_matched_by_queries": popularity_df.count(),
                "total_query_matches": total_popularity,
            },
            "importance": {
                "final_score": importance,
                "coverage_percentage": round(importance * 100, 4),
            },
            "runtimes_sec": {"total": round(runtime, 2)},
        },
        output_file,
    )

    return selected_ids, importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Method 1: popularity-based tuple selection")
    parser.add_argument("--input", required=True, help="CSV file containing the table R")
    parser.add_argument("--queries", required=True, help="CSV file containing the query workload Q")
    parser.add_argument("--threshold", type=int, required=True, help="T, the number of tuples to keep")
    parser.add_argument("--output", required=True, help="CSV file to write R' to")
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip writing the extra *_with_values.csv file",
    )
    args = parser.parse_args()

    run(args.input, args.queries, args.threshold, args.output, save_scores=not args.no_scores)


if __name__ == "__main__":
    main()
