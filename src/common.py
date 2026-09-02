"""Shared helpers: Spark session setup, dataset loading, and output writing.

All three methods load the input table the same way and write their results the
same way, so that logic lives here instead of being duplicated three times.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile
from typing import Iterable, List

# Must be set before PySpark starts. Method 2 shuffles Python tuples through
# partitionBy(), and Spark refuses to do that unless string hashing is
# deterministic across driver and executors — otherwise the same key could land
# in different partitions and the join would silently lose rows.
os.environ.setdefault("PYTHONHASHSEED", "0")

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def get_spark(app_name: str = "dis-valuable-subset", ui_port: str = "4050") -> SparkSession:
    """Return a local SparkSession.

    Adjust ``spark.driver.memory`` here if you hit OOM on the exhaustive
    variants of Method 2 and Method 3.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.ui.port", ui_port)
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.executorEnv.PYTHONHASHSEED", os.environ["PYTHONHASHSEED"])
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    # Spark's INFO chatter drowns out the progress the methods print themselves.
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_table(spark: SparkSession, input_file: str) -> DataFrame:
    """Read the relation R from CSV and attach a stable ``row_id``.

    Every column is cast to integer (the project assumes an all-integer table).
    ``row_id`` is assigned by ordering on all columns, so the same input file
    always produces the same IDs regardless of partitioning — which matters
    because results are compared across methods and across runs.
    """
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    for column in df.columns:
        df = df.withColumn(column, F.col(column).cast(IntegerType()))

    ordering = Window.orderBy(*df.columns)
    return df.withColumn("row_id", F.row_number().over(ordering) - 1).cache()


def write_single_csv(df: DataFrame, output_file: str) -> None:
    """Write ``df`` to exactly one CSV file at ``output_file``.

    Spark writes a directory of part-files; the project asks for a single named
    output file, so we coalesce to one partition, write to a temp directory,
    then move the part-file into place.
    """
    output_file = os.path.abspath(output_file)
    parent = os.path.dirname(output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix=".spark_out_", dir=parent or ".")
    try:
        staging = os.path.join(tmp_dir, "out")
        df.coalesce(1).write.csv(staging, header=True, mode="overwrite")
        parts = glob.glob(os.path.join(staging, "part-*.csv"))
        if not parts:
            raise RuntimeError(f"Spark produced no output part-file in {staging}")
        shutil.move(parts[0], output_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def save_selected(df: DataFrame, selected_ids: Iterable[int], output_file: str) -> None:
    """Write the tuples of R' to ``output_file`` in the input file's format."""
    selected_ids = list(selected_ids)
    if not selected_ids:
        print("No tuples selected; nothing written.")
        return

    data_cols = [c for c in df.columns if c != "row_id"]
    result = df.filter(F.col("row_id").isin(selected_ids)).select(*data_cols)
    write_single_csv(result, output_file)
    print(f"Wrote {len(selected_ids)} tuples to {output_file}")


def save_selected_with_scores(
    df: DataFrame, selected_ids: Iterable[int], scores_df: DataFrame, output_file: str
) -> None:
    """Write R' joined with the per-tuple score columns, for inspection."""
    selected_ids = list(selected_ids)
    if not selected_ids:
        return

    selected = df.filter(F.col("row_id").isin(selected_ids))
    write_single_csv(selected.join(scores_df, "row_id", "inner"), output_file)
    print(f"Wrote scored tuples to {output_file}")


def parse_dataset_name(path: str) -> dict:
    """Pull distribution / row count / column count out of a generated filename.

    Files produced by ``data_generator.py`` are named
    ``data_<distribution>_<rows>[k]_<cols>.csv``. Returns ``None`` values for
    inputs that don't follow the convention — the methods still run, the run log
    is just less informative.
    """
    name = os.path.basename(path)
    match = re.match(r"^data_([A-Za-z0-9_]+)_(\d+)([kK])?_(\d+)\.csv$", name)
    if not match:
        return {"filename": name, "distribution": None, "num_rows": None, "num_cols": None}

    distribution, rows, k_flag, cols = match.groups()
    return {
        "filename": name,
        "distribution": distribution,
        "num_rows": int(rows) * (1000 if k_flag else 1),
        "num_cols": int(cols),
    }


def write_log(log: dict, output_file: str) -> None:
    """Write a run log next to the results as ``<output>.json``."""
    log_path = os.path.splitext(output_file)[0] + ".json"
    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "w") as handle:
        json.dump(log, handle, indent=2)
    print(f"Wrote run log to {log_path}")


def jaccard_distance(a_values: List[int], b_values: List[int]) -> float:
    """Jaccard distance between two tuples treated as sets of (column, value).

    Two tuples over D columns share ``matches`` attribute values, so their
    union has ``2D - matches`` distinct (column, value) members. Distance is
    ``1 - matches / (2D - matches)``: 0 for identical tuples, 1 when no
    attribute agrees.
    """
    num_cols = len(a_values)
    matches = sum(1 for x, y in zip(a_values, b_values) if x == y)
    union = 2 * num_cols - matches
    if union == 0:
        return 0.0
    return 1.0 - (matches / float(union))
