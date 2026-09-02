# Finding the Most Valuable Parts of a Dataset

PySpark implementations of three methods for choosing which tuples to keep when
a relation no longer fits within a storage budget.

Given a table `R`, a query workload `Q`, and a threshold `T`, find the subset
`R' ⊆ R` with `|R'| ≤ T` that maximises `imp(R')`. The interesting part is that
each method defines `imp` differently, and those definitions have wildly
different computational consequences: one is solvable by a sort, one is
NP-hard-flavoured combinatorial search, and one is a Shapley value over `2^n`
coalitions.

Built for the Data Intensive Systems course at Utrecht University (2025–26).

---

## The three methods

### Method 1 — popularity

```
imp(t)  = pop(t)                                    # queries whose answer contains t
imp(R') = Σ_{t∈R'} pop(t) / Σ_{t∈R} pop(t)
```

Tuple importances are independent, so the optimum is just the top `T` by
popularity — no subset search required. The work is in computing popularity for
the whole workload at once: rather than running `|Q|` separate filters, queries
are exploded into one row per predicate, tuples are reshaped into
column→value maps, and a single join plus a "matched predicates == k" filter
evaluates every query simultaneously.

Scales to hundreds of thousands of rows.

### Method 2 — popularity weighted by diversity

```
imp(t)  = pop(t) · mean_{u∈R', u≠t} (1 − sim(t,u))
imp(R') = Σ_{t∈R'} imp(t) / Σ_{t∈R} imp(t)
```

A popular tuple that duplicates other kept tuples is redundant, so popularity is
scaled by average Jaccard distance to the rest of the subset. Now a tuple's
value depends on what else is kept, the objective is no longer separable, and
the top-`T`-by-score shortcut disappears. Three solvers:

| Solver | Approach | Complexity | Practical ceiling |
|---|---|---|---|
| `exact-enum` | Score every `C(n,T)` subset in Spark batches | `O(C(n,T)·T²)` | ~30–50 rows |
| `exact-greedy` | All pairwise distances + incremental marginal gain | `O(n² + T·n)` | ~10k rows |
| `lsh-greedy` | MinHash LSH for near pairs, distance 1.0 otherwise | `O(n·k_LSH + T·n)` | 50k+ rows |

The greedy solvers maintain two running scores per candidate so each iteration
is `O(|remaining|)` rather than a full rescore: `A(s)` accumulates distance from
`s` to already-selected tuples, `B(s)` accumulates `pop(u)·d(s,u)` over those
tuples, and the marginal gain of adding `s` is `B(s) + pop(s)·A(s)`.

### Method 3 — Shapley residual importance

```
imp(R')   = mean_{t∈R'} (1 − sim(t, centroid(R')))
resimp(t) = Σ_{S ⊆ R\{t}}  |S|!(|R|−|S|−1)!/|R|!  ·  [imp(S∪{t}) − imp(S)]
```

Set importance is spread around the centroid, and a tuple's value is its Shapley
value with respect to that set function — its average marginal contribution over
every coalition. This method ignores the query workload entirely; it is purely
about heterogeneity.

`2^n` subsets makes this exponential and it caps out around 25 rows. Three
things keep it as fast as it can be: factorials precomputed once, centroids
updated incrementally in `O(D)` instead of recomputed in `O(|S|·D)`, and subsets
enumerated level-by-level in bounded batches so memory never blows up.

---

## Setup

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs a JDK (8, 11, 17 or 21) on `PATH` for Spark. Everything runs on a local
Spark master; no cluster required.

---

## Usage

Generate a table, generate a workload for it, then run a method. Each method
takes an input table, a threshold, and an output path, and writes `R'` as a
single CSV plus a JSON run log beside it.

```bash
# 1. tables: iid_uniform, popular_values, clustered
python -m src.data_generator --rows 100 --cols 5 --outdir data

# 2. queries for one of them
python -m src.query_generator \
    --data data/data_iid_uniform_100_5.csv \
    --num-queries 1000 --outdir queries

# 3a. Method 1
python -m src.method1 \
    --input data/data_iid_uniform_100_5.csv \
    --queries queries/query_1000_iid_uniform_100_5.csv \
    --threshold 15 --output results/method1.csv

# 3b. Method 2 (swap --algorithm for exact-greedy / lsh-greedy)
python -m src.method2 \
    --input data/data_iid_uniform_100_5.csv \
    --queries queries/query_1000_iid_uniform_100_5.csv \
    --threshold 5 --algorithm exact-enum --output results/method2.csv

# 3c. Method 3 (no queries — diversity only; keep n small)
python -m src.method3 \
    --input data/data_iid_uniform_25_5.csv \
    --threshold 8 --output results/method3.csv
```

`--help` on any of them lists the remaining knobs (`--batch-size`,
`--lsh-threshold`, `--domain-size`, `--seed`, …). Set `SPARK_DRIVER_MEMORY=8g`
if the exhaustive solvers run out of heap.

Each run writes:

- `results/<name>.csv` — the selected tuples, same schema as the input
- `results/<name>_with_values.csv` — the same tuples plus their scores
- `results/<name>.json` — parameters, importance, and a per-phase runtime breakdown

---

## Datasets

No dataset is provided with the problem, so the generator produces three shapes
chosen to stress the methods differently:

- **`iid_uniform`** — every value equally likely. Tuples are mutually distant and
  queries match few rows. The baseline.
- **`popular_values`** — a configurable fraction of columns is Zipf-skewed, so a
  few values dominate. Tuples become similar and popularity concentrates. This
  is the hard case for LSH (many near-neighbour pairs to store) and the one that
  produces the lowest diversity scores.
- **`clustered`** — rows drawn around a few per-column centres with Gaussian
  noise, in geometrically-decaying group sizes. Dense redundant groups, so this
  is where diversity-aware selection visibly beats pure popularity: it picks one
  representative per cluster.

The query generator matters as much as the data generator. Uniformly random
predicates would give nearly every tuple popularity 0 or 1, making popularity
useless as a signal. Instead, predicate values are sampled from each column's
empirical distribution with an adjustable bias towards frequent values, and a
pool of reusable "popular patterns" is shared across roughly 35% of the
workload so the same tuples get returned repeatedly.

---

## Selected results

From the accompanying report (Google Colab, 1–2 Spark workers — a real cluster
would do considerably better on the parallel parts).

**Method 1 scales; the exact methods do not.**

| Method | Largest input handled | Runtime there |
|---|---|---|
| Method 1 | 750,000 rows | ~32 min |
| Method 2 `lsh-greedy` | 50,000 rows | 62 s |
| Method 2 `exact-greedy` | 10,000 rows | 95 min |
| Method 2 `exact-enum` | 100 rows (T=5) | 189 min |
| Method 3 | 25 rows | 5.6 h |

**Approximation is nearly free.** On IID-uniform data, exact-greedy vs
LSH-greedy total runtime:

| Rows | `exact-greedy` | `lsh-greedy` | Speedup |
|---:|---:|---:|---:|
| 1,000 | 86 s | 32 s | 2.7× |
| 5,000 | 1,092 s | 26 s | 41× |
| 10,000 | 5,690 s | 29 s | 196× |
| 50,000 | crashed (OOM) | 62 s | — |

Exact-greedy's precompute grows quadratically (0.5 s → 66 s from 100 to 10,000
rows) while LSH's stays flat (0.9 s → 4.5 s), because LSH only ever materialises
near-neighbour pairs.

On small inputs where exhaustive enumeration is feasible, the greedy solvers
recover the optimum or come within a fraction of a percent of it. A reproducible
example on a 40-row IID-uniform table with T=5 and 300 queries: `exact-enum`
scores `0.393650` after evaluating 658,008 subsets; `exact-greedy` reaches the
identical `0.393650` in 0.46 s of selection; `lsh-greedy` gets `0.385159`.

**Data distribution matters more than you'd expect.** Clustered data yields the
highest importance scores across all Method 2 solvers, since the selector can
pick one representative per cluster. Popular-values data yields the lowest, and
is also 7× slower under LSH (208 s vs 29 s at 10,000 rows) because so many
tuples hash into the same buckets that the near-pair table explodes.

---

## Correctness checks

`Σ_t resimp(t)` should equal `imp(R)` — the Shapley efficiency axiom, since the
grand coalition's value gets fully distributed among its members. On a 12×5
IID-uniform table:

```
sum of residual importances : 0.9444444444444444
imp(R)                      : 0.9444444444444443
```

Agreement to floating-point error. This is a genuinely useful test, because it
catches weighting mistakes that a plausible-looking ranking would hide.

For Method 2, `exact-enum` is the ground truth the greedy solvers are checked
against on inputs small enough for both to run.

---

## Repo layout

```
src/
  common.py           Spark session, table loading, single-file CSV output, Jaccard distance
  data_generator.py   Synthetic tables: iid_uniform / popular_values / clustered
  query_generator.py  Query workloads with controlled skew and overlap
  method1.py          Popularity-based selection
  method2.py          Popularity × diversity, three solvers
  method3.py          Shapley residual importance
data/                 Generated tables (gitignored — regenerate with the CLI)
queries/              Generated workloads (gitignored)
results/              Selected subsets and run logs (gitignored)
```

Generated CSVs are gitignored rather than committed. Everything is seeded, so
`--seed 42` reproduces the exact tables and workloads used in the experiments.

---

## Notes on this version

This repository is a cleaned-up version of code originally written as Colab
notebooks. Beyond removing Colab-specific setup and hardcoded Drive paths, two
substantive changes:

- **Method 3's Shapley weight was being applied twice** (`weight * (weight * Δ)`),
  which squared the weights and distorted the relative contribution of different
  coalition sizes. Fixed here; the efficiency check above now passes, which it
  would not have before. Residual importance values from this version therefore
  differ from those in the original report, though the runtime figures — which
  are what the report's scaling analysis rests on — are unaffected.
- Methods now write a **single named CSV** rather than a Spark part-file
  directory, matching the problem statement's "name of the output file".

## Authors

Deniz Cangı, Bas J. Cornelissen, Nadin Harshuk — Utrecht University.
