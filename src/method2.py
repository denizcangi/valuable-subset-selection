"""Method 2 — popularity weighted by dissimilarity to the rest of the subset.

A tuple that is popular but nearly identical to other kept tuples is redundant,
so its popularity is scaled by its average Jaccard distance to the rest of R':

    imp(t)  = pop(t) * mean_{u in R', u != t} (1 - sim(t, u))
    imp(R') = sum_{t in R'} imp(t) / sum_{t in R} imp(t)

Now a tuple's value depends on which other tuples are kept, so the objective is
no longer separable and the optimum is not just "the top T by score". Three
algorithms are implemented:

  exact-enum    Evaluate every subset of size T. Optimal, but C(n, T) subsets
                makes it infeasible past a few dozen candidate tuples.
  exact-greedy  Precompute all pairwise distances, then add tuples one at a
                time by marginal gain. O(n^2) precompute dominates.
  lsh-greedy    Same greedy loop, but MinHash LSH supplies distances only for
                pairs closer than a cutoff; everything else is treated as
                distance 1.0. Avoids the O(n^2) blow-up.

Run:
    python -m src.method2 --input data/data_iid_uniform_25_5.csv \\
        --queries queries/query_1000_iid_uniform_25_5.csv \\
        --threshold 5 --algorithm exact-enum --output results/method2.csv
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from itertools import combinations, islice
from math import comb
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.storagelevel import StorageLevel

from .common import (
    get_spark,
    jaccard_distance,
    load_table,
    parse_dataset_name,
    save_selected,
    save_selected_with_scores,
    write_log,
)
from .method1 import Method1

IMPORTANCE_SCHEMA = StructType(
    [
        StructField("row_id", IntegerType(), False),
        StructField("popularity", IntegerType(), False),
        StructField("avg_dissimilarity", DoubleType(), False),
        StructField("importance", DoubleType(), False),
    ]
)


class Method2:
    """Popularity-and-diversity selection with exact and approximate solvers."""

    def __init__(self, spark: SparkSession, input_file: str):
        self.spark = spark
        self.input_file = input_file
        self.df = load_table(spark, input_file)
        self.num_rows = self.df.count()
        self.data_cols = [c for c in self.df.columns if c != "row_id"]

        # Run metadata, filled in by whichever solver is invoked.
        self.algorithm = None
        self.threshold = None
        self.query_file_path = None
        self.lsh_threshold = None
        self.batch_size = None
        self.num_candidates = 0
        self.total_combinations = None
        self.tuples_with_matches = 0

        self.selected_ids: List[int] = []
        self.importance_of_selected = 0.0
        self.importance_of_all = 0.0
        self.final_score = 0.0
        self.importance_data: List[dict] = []
        self.importance_df: DataFrame | None = None

        self.runtime_popularity = 0.0
        self.runtime_precompute = 0.0
        self.runtime_greedy = 0.0
        self.runtime_enumeration = 0.0
        self.runtime_denominator = 0.0
        self.runtime_total = 0.0

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------

    def compute_popularity(self, query_file_path: str) -> DataFrame:
        """Reuse Method 1's distributed join to get per-tuple popularity."""
        popularity_df, _ = Method1(self.spark, self.input_file).compute_popularity(
            query_file_path
        )
        return popularity_df.cache()

    def compute_exact_pairwise_distances(self, candidates_df: DataFrame) -> DataFrame:
        """All pairwise Jaccard distances, computed inside Spark.

        A self cross-join filtered to ``id1 < id2`` gives each unordered pair
        once. The match count is built as a sum of per-column CASE expressions,
        so the whole thing stays a single Spark plan rather than a Python loop.
        """
        start = time.time()
        left = candidates_df.alias("a")
        right = candidates_df.alias("b")
        pairs = left.crossJoin(right).where(F.col("a.row_id") < F.col("b.row_id"))

        matches_expr = sum(
            F.when(F.col(f"a.{c}") == F.col(f"b.{c}"), F.lit(1)).otherwise(F.lit(0))
            for c in self.data_cols
        )
        num_cols = float(len(self.data_cols))

        distances = (
            pairs.select(
                F.col("a.row_id").alias("id1"),
                F.col("b.row_id").alias("id2"),
                matches_expr.alias("matches"),
            )
            .withColumn("union_sz", F.lit(2.0 * num_cols) - F.col("matches").cast("double"))
            .withColumn("jacc_sim", F.col("matches").cast("double") / F.col("union_sz"))
            .withColumn("dissimilarity", F.lit(1.0) - F.col("jacc_sim"))
            .select("id1", "id2", "dissimilarity")
        ).cache()

        count = distances.count()
        print(f"Computed {count} exact pairwise distances in {time.time() - start:.2f}s")
        return distances

    def _build_adjacency_rdd(self, distances_df: DataFrame):
        """Turn the (id1 < id2) distance table into a partitioned adjacency RDD.

        Pairs are stored one way round only, so both directions are unioned
        before building ``row_id -> (neighbour, distance)``. Partitioning by key
        makes ``lookup()`` hit a single partition instead of scanning
        everything, which is what the greedy loop needs on each iteration.
        """
        symmetric = distances_df.select(
            F.col("id1").alias("src"), F.col("id2").alias("dst"), F.col("dissimilarity").alias("dist")
        ).unionByName(
            distances_df.select(
                F.col("id2").alias("src"),
                F.col("id1").alias("dst"),
                F.col("dissimilarity").alias("dist"),
            )
        )

        num_partitions = min(64, max(self.spark.sparkContext.defaultParallelism * 4, 16))
        adjacency = (
            symmetric.select("src", "dst", "dist")
            .rdd.map(lambda r: (int(r["src"]), (int(r["dst"]), float(r["dist"]))))
            .partitionBy(num_partitions)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        adjacency.count()
        return adjacency

    def _build_features(self, rows_df: DataFrame, num_features: int = 1 << 18) -> DataFrame:
        """Encode each tuple as a binary sparse vector for MinHash LSH.

        A tuple (A0=5, A1=10) becomes the token set {"A0:5", "A1:10"}, which
        HashingTF maps to a binary vector. MinHash over that vector estimates
        exactly the Jaccard similarity our distance function uses.
        """
        tokens = F.array(
            *[F.concat(F.lit(c + ":"), F.col(c).cast(StringType())) for c in self.data_cols]
        )
        tokenised = rows_df.select("row_id", tokens.alias("tokens"))
        hashing_tf = HashingTF(
            inputCol="tokens", outputCol="features", binary=True, numFeatures=num_features
        )
        return hashing_tf.transform(tokenised).select("row_id", "features").cache()

    def compute_denominator_exact(self, pop_df: DataFrame, distances_df: DataFrame) -> float:
        """imp(R) — the normalising denominator — from exact distances."""
        r_pop = self.df.join(pop_df, "row_id", "inner").select("row_id", "popularity")

        forward = distances_df.select(
            F.col("id1").alias("a"), F.col("dissimilarity").alias("dist")
        )
        reverse = distances_df.select(
            F.col("id2").alias("a"), F.col("dissimilarity").alias("dist")
        )
        totals = (
            forward.unionByName(reverse)
            .groupBy("a")
            .agg(F.sum("dist").alias("total_dist"))
        )

        divisor = float(max(1, r_pop.count() - 1))
        result = (
            r_pop.join(totals, r_pop.row_id == totals.a, "inner")
            .select(r_pop.popularity.alias("pop"), F.col("total_dist"))
            .withColumn("avg_dist", F.col("total_dist") / F.lit(divisor))
            .withColumn("term", F.col("pop") * F.col("avg_dist"))
        )

        value = result.agg(F.sum("term").alias("denom")).first()["denom"]
        return float(value) if value is not None else 0.0

    def compute_denominator_lsh(self, pop_df: DataFrame, near_pairs: DataFrame) -> float:
        """imp(R) estimated from LSH near-pairs only.

        LSH gives exact distances for pairs below the cutoff and says nothing
        about the rest. Anything not reported is at least as far as the cutoff,
        so it is approximated as distance 1.0 — the same fallback the greedy
        loop uses, which keeps numerator and denominator consistent.
        """
        r_pop = self.df.join(pop_df, "row_id", "inner").select("row_id", "popularity")

        near_agg = near_pairs.select(
            F.col("id1").alias("a"), F.col("dissimilarity").alias("dist_bw")
        ).groupBy("a").agg(
            F.count("*").alias("close_count"),
            F.sum("dist_bw").alias("close_dist"),
        )

        others = float(self.num_rows - 1)
        result = (
            r_pop.join(near_agg, r_pop.row_id == near_agg.a, "left")
            .select(
                r_pop.popularity.alias("pop"),
                F.coalesce("close_count", F.lit(0)).alias("close_count"),
                F.coalesce("close_dist", F.lit(0.0)).alias("close_dist"),
            )
            .withColumn(
                "sum_distances",
                F.col("close_dist") + (F.lit(others) - F.col("close_count")) * F.lit(1.0),
            )
            .withColumn("avg_distances", F.col("sum_distances") / F.lit(others))
            .withColumn("term", F.col("pop") * F.col("avg_distances"))
        )

        value = result.agg(F.sum("term").alias("denom")).first()["denom"]
        return float(value) if value is not None else 0.0

    # ------------------------------------------------------------------
    # Solver 1: exhaustive subset enumeration
    # ------------------------------------------------------------------

    def exact_enumeration(
        self, threshold: int, query_file_path: str, batch_size: int = 10000
    ) -> Tuple[List[int], float, DataFrame]:
        """Evaluate every subset of size T and keep the best. Optimal, slow.

        Subsets are generated lazily and pushed to Spark in batches so the
        driver never materialises C(n, T) combinations at once. Each batch is
        scored in parallel and reduced to its best member; only the running
        global best is retained between batches.
        """
        self.algorithm = "Method2-Exact-Subset-Enumeration"
        self.threshold = threshold
        self.query_file_path = query_file_path
        self.batch_size = batch_size

        start_total = time.time()

        start = time.time()
        pop_df = self.compute_popularity(query_file_path)
        self.runtime_popularity = time.time() - start
        self.tuples_with_matches = pop_df.count()
        print(f"Popularity computed in {self.runtime_popularity:.2f}s")

        if self.tuples_with_matches == 0 or threshold <= 0:
            return self._empty_result()

        # Tuples no query returns have pop = 0 and therefore contribute nothing
        # to the numerator, so they can be dropped from the search space.
        candidates = (
            self.df.join(pop_df, "row_id", "inner")
            .select("row_id", "popularity", *self.data_cols)
            .filter(F.col("popularity") > 0)
            .cache()
        )
        candidate_ids = [r["row_id"] for r in candidates.select("row_id").collect()]
        pop_map: Dict[int, int] = {
            r["row_id"]: int(r["popularity"])
            for r in candidates.select("row_id", "popularity").collect()
        }
        print(f"Candidate tuples: {len(candidate_ids)}")

        threshold = min(threshold, len(candidate_ids))
        self.num_candidates = len(candidate_ids)

        start = time.time()
        distances_df = self.compute_exact_pairwise_distances(
            candidates.select("row_id", *self.data_cols)
        )
        adjacency = self._build_adjacency_rdd(distances_df)
        self.runtime_precompute = time.time() - start

        # Workers need random access to distances, so collect the adjacency
        # structure into a nested dict and broadcast it once.
        distance_map = {
            row_a: dict(neighbours)
            for row_a, neighbours in adjacency.groupByKey().mapValues(list).collect()
        }
        bc_distances = self.spark.sparkContext.broadcast(distance_map)
        bc_pop = self.spark.sparkContext.broadcast(pop_map)

        start = time.time()
        self.importance_of_all = self.compute_denominator_exact(pop_df, distances_df)
        self.runtime_denominator = time.time() - start

        total_subsets = comb(len(candidate_ids), threshold)
        self.total_combinations = total_subsets
        print(f"Evaluating {total_subsets:,} subsets in batches of {batch_size:,}")

        def score_subset(subset):
            """Numerator of imp(R') for one candidate subset."""
            ids = list(subset)
            pop = bc_pop.value
            distances = bc_distances.value

            if len(ids) == 1:
                return (tuple(ids), float(pop.get(ids[0], 0)))

            divisor = float(len(ids) - 1)
            total = 0.0
            for t in ids:
                pop_t = float(pop.get(t, 0))
                if pop_t == 0.0:
                    continue
                neighbours = distances.get(t, {})
                # Missing entries only occur under the LSH variant; here the
                # map is complete, but the default keeps the two paths aligned.
                dist_sum = sum(neighbours.get(u, 1.0) for u in ids if u != t)
                total += pop_t * (dist_sum / divisor)
            return (tuple(ids), total)

        start = time.time()
        best_subset, best_score = None, float("-inf")
        cores = self.spark.sparkContext.defaultParallelism
        subsets = combinations(candidate_ids, threshold)

        while True:
            batch = list(islice(subsets, batch_size))
            if not batch:
                break
            num_slices = min(len(batch), max(cores * 2, 16))
            rdd = self.spark.sparkContext.parallelize(batch, numSlices=num_slices)
            batch_best = rdd.map(score_subset).max(key=lambda x: x[1])
            if batch_best[1] > best_score:
                best_score = float(batch_best[1])
                best_subset = list(batch_best[0])

        self.runtime_enumeration = time.time() - start
        print(f"Enumeration took {self.runtime_enumeration:.2f}s")

        self.selected_ids = best_subset or []
        self.importance_of_selected = best_score if best_subset else 0.0
        self._build_importance_table(pop_map, distance_map)
        self._finalise(start_total)

        candidates.unpersist()
        distances_df.unpersist()
        adjacency.unpersist()
        bc_distances.unpersist()
        bc_pop.unpersist()
        pop_df.unpersist()
        distance_map.clear()
        gc.collect()

        return self.selected_ids, self.final_score, self.importance_df

    # ------------------------------------------------------------------
    # Solver 2: greedy with exact distances
    # ------------------------------------------------------------------

    def greedy_exact(
        self, threshold: int, query_file_path: str
    ) -> Tuple[List[int], float, DataFrame]:
        """Greedy selection using exact pairwise distances.

        The marginal gain of adding s to the current R' is

            score(s) = B(s) + pop(s) * A(s)

        where A(s) is the summed distance from s to already-selected tuples and
        B(s) is the summed ``pop(u) * d(s, u)`` over those tuples — the benefit
        the already-selected tuples get from s. Both are maintained
        incrementally, so each iteration costs O(|remaining|) instead of
        rescoring every subset from scratch.
        """
        self.algorithm = "Method2-Exact-Greedy"
        self.threshold = threshold
        self.query_file_path = query_file_path

        start_total = time.time()

        start = time.time()
        pop_df = self.compute_popularity(query_file_path)
        self.runtime_popularity = time.time() - start
        self.tuples_with_matches = pop_df.count()
        print(f"Popularity computed in {self.runtime_popularity:.2f}s")

        if self.tuples_with_matches == 0 or threshold <= 0:
            return self._empty_result()

        candidates = (
            self.df.join(pop_df, "row_id", "inner")
            .select("row_id", "popularity", *self.data_cols)
            .cache()
        )
        print(f"Candidate tuples: {candidates.count()}")

        start = time.time()
        distances_df = self.compute_exact_pairwise_distances(
            candidates.select("row_id", *self.data_cols)
        )
        adjacency = self._build_adjacency_rdd(distances_df)
        self.runtime_precompute = time.time() - start

        pop_map: Dict[int, int] = {
            r["row_id"]: int(r["popularity"])
            for r in candidates.select("row_id", "popularity").collect()
        }
        self.num_candidates = len(pop_map)

        start = time.time()
        remaining = set(pop_map)
        a_scores = {cid: 0.0 for cid in pop_map}
        b_scores = {cid: 0.0 for cid in pop_map}
        neighbour_cache: Dict[int, Dict[int, float]] = {}

        for i in range(min(threshold, len(remaining))):
            if i == 0:
                # All gains are zero on the first pick, so it reduces to
                # picking the single most popular tuple.
                best_id = max(remaining, key=lambda t: pop_map[t])
            else:
                best_id = max(remaining, key=lambda s: b_scores[s] + pop_map[s] * a_scores[s])

            self.selected_ids.append(best_id)
            remaining.remove(best_id)

            if best_id not in neighbour_cache:
                neighbour_cache[best_id] = dict(adjacency.lookup(int(best_id)))

            if remaining:
                dist_map = neighbour_cache[best_id]
                best_pop = pop_map[best_id]
                for s in remaining:
                    dist = dist_map.get(s, 1.0)
                    a_scores[s] += dist
                    b_scores[s] += best_pop * dist

        self.runtime_greedy = time.time() - start
        print(f"Greedy selection took {self.runtime_greedy:.2f}s")

        self._build_importance_table(pop_map, neighbour_cache)
        self.importance_of_selected = sum(row["importance"] for row in self.importance_data)

        start = time.time()
        self.importance_of_all = self.compute_denominator_exact(pop_df, distances_df)
        self.runtime_denominator = time.time() - start
        self._finalise(start_total)

        candidates.unpersist()
        distances_df.unpersist()
        adjacency.unpersist()
        pop_df.unpersist()

        return self.selected_ids, self.final_score, self.importance_df

    # ------------------------------------------------------------------
    # Solver 3: greedy with LSH-approximated distances
    # ------------------------------------------------------------------

    def greedy_lsh(
        self, threshold: int, query_file_path: str, lsh_threshold: float = 0.8
    ) -> Tuple[List[int], float, DataFrame]:
        """Greedy selection using MinHash LSH instead of all pairwise distances.

        ``approxSimilarityJoin`` returns only pairs within ``lsh_threshold``
        Jaccard distance of each other. Everything else is assumed to be at
        distance 1.0. Since the objective rewards distant tuples and most pairs
        in a large table really are near-maximally distant, the approximation
        costs little accuracy while removing the O(n^2) precompute.
        """
        self.algorithm = "Method2-LSH-Greedy"
        self.threshold = threshold
        self.query_file_path = query_file_path
        self.lsh_threshold = lsh_threshold

        start_total = time.time()

        start = time.time()
        pop_df = self.compute_popularity(query_file_path)
        self.runtime_popularity = time.time() - start
        self.tuples_with_matches = pop_df.count()
        print(f"Popularity computed in {self.runtime_popularity:.2f}s")

        if self.tuples_with_matches == 0 or threshold <= 0:
            return self._empty_result()

        candidates = self.df.join(pop_df, "row_id", "inner").select("row_id", "popularity").cache()
        self.num_candidates = candidates.count()

        start = time.time()
        features = self._build_features(
            candidates.join(self.df, "row_id", "inner").select("row_id", *self.data_cols)
        )
        lsh_model = MinHashLSH(
            inputCol="features", outputCol="hashes", numHashTables=5
        ).fit(features)

        near_pairs = (
            lsh_model.approxSimilarityJoin(
                features.alias("A"), features.alias("B"), lsh_threshold, distCol="distance"
            )
            .select(
                F.col("datasetA.row_id").alias("id1"),
                F.col("datasetB.row_id").alias("id2"),
                F.col("distance").alias("dissimilarity"),
            )
            .filter(F.col("id1") != F.col("id2"))
            .cache()
        )

        near_list = near_pairs.collect()
        dissimilarity: Dict[Tuple[int, int], float] = {}
        for row in near_list:
            key = (row["id1"], row["id2"])
            dissimilarity[key] = row["dissimilarity"]
            dissimilarity[(key[1], key[0])] = row["dissimilarity"]
        self.runtime_precompute = time.time() - start
        print(f"Found {len(near_list)} near pairs in {self.runtime_precompute:.2f}s")

        pop_map: Dict[int, int] = {
            r["row_id"]: int(r["popularity"]) for r in candidates.collect()
        }

        def get_distance(t: int, u: int) -> float:
            return dissimilarity.get((t, u), 1.0)

        start = time.time()
        remaining = set(pop_map)
        a_scores = {cid: 0.0 for cid in pop_map}
        b_scores = {cid: 0.0 for cid in pop_map}

        for i in range(min(threshold, len(remaining))):
            if i == 0:
                best_id = max(remaining, key=lambda t: pop_map[t])
            else:
                best_id = max(remaining, key=lambda s: b_scores[s] + pop_map[s] * a_scores[s])

            self.selected_ids.append(best_id)
            remaining.remove(best_id)

            best_pop = pop_map[best_id]
            for s in remaining:
                dist = get_distance(s, best_id)
                a_scores[s] += dist
                b_scores[s] += best_pop * dist

        self.runtime_greedy = time.time() - start
        print(f"Greedy selection took {self.runtime_greedy:.2f}s")

        # The selected set is small, so score it with exact distances even
        # though selection used approximations — this keeps the reported
        # importance honest rather than inheriting LSH's error.
        selected_rows = (
            self.df.filter(F.col("row_id").isin(self.selected_ids))
            .select("row_id", *self.data_cols)
            .collect()
        )
        values = {r["row_id"]: [r[c] for c in self.data_cols] for r in selected_rows}
        exact_neighbours = {
            t: {u: jaccard_distance(values[t], values[u]) for u in self.selected_ids if u != t}
            for t in self.selected_ids
        }

        self._build_importance_table(pop_map, exact_neighbours)
        self.importance_of_selected = sum(row["importance"] for row in self.importance_data)

        start = time.time()
        self.importance_of_all = self.compute_denominator_lsh(pop_df, near_pairs)
        self.runtime_denominator = time.time() - start
        self._finalise(start_total)

        candidates.unpersist()
        features.unpersist()
        near_pairs.unpersist()
        pop_df.unpersist()
        dissimilarity.clear()

        return self.selected_ids, self.final_score, self.importance_df

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------

    def _build_importance_table(self, pop_map, neighbours) -> None:
        """Per-tuple popularity, average dissimilarity and importance in R'."""
        self.importance_data = []
        if not self.selected_ids:
            self.importance_df = self.spark.createDataFrame([], schema=IMPORTANCE_SCHEMA)
            return

        if len(self.selected_ids) == 1:
            t = self.selected_ids[0]
            pop_t = int(pop_map.get(t, 0))
            self.importance_data.append(
                {
                    "row_id": int(t),
                    "popularity": pop_t,
                    "avg_dissimilarity": 0.0,
                    "importance": float(pop_t),
                }
            )
        else:
            divisor = float(len(self.selected_ids) - 1)
            for t in self.selected_ids:
                pop_t = int(pop_map.get(t, 0))
                dist_map = neighbours.get(t, {})
                dist_sum = sum(dist_map.get(u, 1.0) for u in self.selected_ids if u != t)
                avg_dist = dist_sum / divisor
                self.importance_data.append(
                    {
                        "row_id": int(t),
                        "popularity": pop_t,
                        "avg_dissimilarity": float(avg_dist),
                        "importance": float(pop_t * avg_dist),
                    }
                )

        self.importance_df = self.spark.createDataFrame(
            self.importance_data, schema=IMPORTANCE_SCHEMA
        )

    def _finalise(self, start_total: float) -> None:
        self.final_score = (
            self.importance_of_selected / self.importance_of_all
            if self.importance_of_all > 0
            else 0.0
        )
        self.runtime_total = time.time() - start_total
        print(f"imp(R') = {self.final_score:.6f}  (total {self.runtime_total:.2f}s)")

    def _empty_result(self):
        print("No tuple is returned by any query; nothing to select.")
        self.importance_df = self.spark.createDataFrame([], schema=IMPORTANCE_SCHEMA)
        return [], 0.0, self.importance_df

    def build_log(self) -> dict:
        query_meta = None
        if self.query_file_path:
            queries = pd.read_csv(self.query_file_path)
            query_meta = {
                "filename": self.query_file_path.split("/")[-1],
                "num_queries": len(queries),
                "avg_k": float(queries["k"].mean()) if "k" in queries else None,
                "avg_selectivity": (
                    float(queries["selectivity"].mean()) if "selectivity" in queries else None
                ),
            }

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": self.algorithm or "Method2",
            "data_file": parse_dataset_name(self.input_file),
            "query_file": query_meta,
            "parameters": {
                "threshold": self.threshold,
                "lsh_threshold": self.lsh_threshold,
                "batch_size": self.batch_size,
                "num_candidates": self.num_candidates,
                "total_combinations": self.total_combinations,
            },
            "results": {
                "tuples_selected": len(self.selected_ids),
                "data_kept_percentage": round(
                    len(self.selected_ids) / self.num_rows * 100, 4
                )
                if self.num_rows
                else 0.0,
                "tuples_matched_by_queries": int(self.tuples_with_matches),
            },
            "importance": {
                "selected": float(self.importance_of_selected),
                "all": float(self.importance_of_all),
                "final_score": float(self.final_score),
                "coverage_percentage": round(self.final_score * 100, 4),
            },
            "runtimes_sec": {
                "popularity": round(self.runtime_popularity, 2),
                "precompute": round(self.runtime_precompute, 2),
                "greedy": round(self.runtime_greedy, 2),
                "enumeration": round(self.runtime_enumeration, 2),
                "denominator": round(self.runtime_denominator, 2),
                "total": round(self.runtime_total, 2),
            },
            "selected_ids": [int(x) for x in self.selected_ids],
            "per_tuple_importance": self.importance_data,
        }


def run(
    input_file: str,
    query_file: str,
    threshold: int,
    output_file: str,
    algorithm: str = "exact-enum",
    lsh_threshold: float = 0.8,
    batch_size: int = 10000,
    save_scores: bool = True,
) -> Tuple[List[int], float]:
    spark = get_spark("method2")
    method = Method2(spark, input_file)

    if algorithm == "exact-enum":
        selected_ids, score, importance_df = method.exact_enumeration(
            threshold, query_file, batch_size=batch_size
        )
    elif algorithm == "exact-greedy":
        selected_ids, score, importance_df = method.greedy_exact(threshold, query_file)
    elif algorithm == "lsh-greedy":
        selected_ids, score, importance_df = method.greedy_lsh(
            threshold, query_file, lsh_threshold=lsh_threshold
        )
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; "
            "choose exact-enum, exact-greedy or lsh-greedy"
        )

    save_selected(method.df, selected_ids, output_file)
    if save_scores and selected_ids:
        save_selected_with_scores(
            method.df, selected_ids, importance_df, output_file.replace(".csv", "_with_values.csv")
        )
    write_log(method.build_log(), output_file)

    return selected_ids, score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method 2: popularity weighted by diversity"
    )
    parser.add_argument("--input", required=True, help="CSV file containing the table R")
    parser.add_argument("--queries", required=True, help="CSV file containing the query workload Q")
    parser.add_argument("--threshold", type=int, required=True, help="T, the number of tuples to keep")
    parser.add_argument("--output", required=True, help="CSV file to write R' to")
    parser.add_argument(
        "--algorithm",
        default="exact-enum",
        choices=["exact-enum", "exact-greedy", "lsh-greedy"],
        help="Solver to use (default: exact-enum, optimal but only for tiny inputs)",
    )
    parser.add_argument(
        "--lsh-threshold",
        type=float,
        default=0.8,
        help="Jaccard distance cutoff for LSH near-pairs (lsh-greedy only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Subsets per Spark batch (exact-enum only)",
    )
    parser.add_argument("--no-scores", action="store_true", help="Skip the *_with_values.csv file")
    args = parser.parse_args()

    run(
        args.input,
        args.queries,
        args.threshold,
        args.output,
        algorithm=args.algorithm,
        lsh_threshold=args.lsh_threshold,
        batch_size=args.batch_size,
        save_scores=not args.no_scores,
    )


if __name__ == "__main__":
    main()
