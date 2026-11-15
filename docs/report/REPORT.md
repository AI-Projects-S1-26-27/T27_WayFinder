# Routes That Fit You

## A Personalised Multi-Objective Walking Route Planner with Learned Search Heuristics

**CS F407 Artificial Intelligence — Final Project Report**

**Team 27 · WayFinder** · Track: **Application** (Search & Problem Solving + Applications of Deep Learning)

Vinny Polamarasetty · Soujanya Lakshmi Tayyala · Chandana N · Chaitanya Srikar Kastury ·
Anisha Potdar · Ananya Avunooru · Anushka Sahoo · Srujana N S

Repository: <https://github.com/ChaitanyaK07/T27_WayFinder>

---

## Abstract

We built a walking route planner for the Jubilee Hills / Banjara Hills area of Hyderabad in which the user, not
the system, decides what matters. Five sliders weight walking time, junction load, traffic exposure, canopy
(tree-cover proximity) and climb. The planner returns the best route under those weights on a turn-aware graph of
13,246 nodes, 34,708 directed edges and 103,384 turn transitions built from OpenStreetMap, ESA WorldCover and the
Copernicus GLO-30 DEM.

We first measured the problem the proposal set out to study. As weight moves away from walking time, the
admissible distance heuristic weakens: A\* expansions rise from 35 % to 81 % of Dijkstra's. We then trained a
preference-conditioned neural heuristic on 8,000,000 exact labels from 8,000 backward Dijkstra runs. It is very
accurate: the mean ratio of predicted to true cost-to-go is 1.02, against 0.17 for the admissible heuristic, and
it is equally accurate on weight vectors never seen in training. Used directly in A\*, it expands 11× fewer states
than A\* with the admissible heuristic, but without any guarantee: routes are 4.4 % worse on average and up to
51 % worse.

Inside Focal Search, which guarantees a route within W × the optimum, the network does not pay off. All 7,000
bounded runs respected their guarantee. However, the learned guide saves only 15 % of expansions against A\* at
W = 1.5, about the same as weighted A\* and a tuned classical guide. Even a perfect guide saves only 18 %. We show
why: FOCAL admits a state only within W × f_min, and f_min is driven by the weak admissible heuristic. We measure
this with a per-query "focal floor". Following the proposal's Week-4 rule, the headline therefore falls back to
Focal Search with a tuned classical guide.

The delivered application returns the exact best route by default. This was verified on 147 random trips, with 0
violations of slider monotonicity in 735 checks. A full pipeline run from a fresh clone of the repository reproduced
every reported result; only wall-clock timings differed. No user study was run, so whether people agree with the
routes is untested (Section 5.3).

---

## 1. Introduction

### 1.1 Motivation

Mainstream route planners answer one question: what is the fastest way there? Many walking trips are not about
that. A student with a heavy bag will accept a few extra minutes to avoid a staircase. Someone recovering from a
knee injury routes around gradients. In a Hyderabad summer, walking under tree cover is worth a detour. These
preferences differ from person to person and from trip to trip, and no widely available tool lets a person state them.

### 1.2 Why this is technically hard

A\* beats Dijkstra by using a heuristic: an estimate of the remaining cost that pushes the search toward the goal.
For plain travel time, straight-line distance is a sound estimate. It never overestimates (admissibility), which
is what makes A\* return optimal routes. Once the cost also includes junction penalties, traffic exposure, canopy and
climb, a distance-based estimate stays admissible but becomes a weak guide, and A\* degenerates toward Dijkstra.
The stronger the user's non-distance preferences, the worse this gets — exactly when personalisation matters most.

### 1.3 Hypotheses

The proposal's goals translate into three testable hypotheses. Section 5.1 gives the verdict on each.

* **H1 — the problem is real.** As the weight on non-distance objectives grows, the admissible distance heuristic
  recovers less of the true cost, and A\* expansions approach Dijkstra's.
* **H2 — the cost-to-go is learnable across preferences.** One network conditioned on the weight vector can predict
  the true cost-to-go accurately, including for weight vectors it never saw in training.
* **H3 — learned guidance speeds up bounded search.** Used as the FOCAL guide in Focal Search, the learned heuristic
  expands fewer states than A\*, weighted A\* and a tuned classical guide at the same suboptimality bound W.

### 1.4 Contributions

1. A **turn-aware, five-objective cost model** on a real street graph, with every feature labelled *measured*,
   *derived* or *proxy* (Sections 3.1–3.2).
2. An **admissible and consistent heuristic** whose consistency is guaranteed by construction, and checked on every
   edge (Section 3.2.3).
3. The **problem stated as a measurement**: expansions against non-distance weight (Section 4.2).
4. A **preference-conditioned learned heuristic**: one network for every slider setting, trained on exact labels from
   backward Dijkstra (Section 3.3).
5. **Focal Search** that uses the learned heuristic for guidance and the admissible heuristic for a provable
   `W ×` bound, with the bound verified for W = 1.0 … 2.0 against exact optima (Sections 3.4, 4.4).
6. An **evaluation** on 200 frozen queries with three training seeds, heuristic-quality analysis and
   generalisation to held-out preferences (Section 4).
7. A **routing service and web interface** with five sliders, verified to return the exact best route and to handle
   bad input cleanly (Sections 3.5, 4.10).

The project makes no novelty claim. Its contribution is a ground-truth-verified evaluation of preference-conditioned
learned heuristics and Focal Search, including a negative result and the measurement that explains it, delivered as
a working application.

---

## 2. Background and related work

### 2.1 Background

**Best-first search.** Dijkstra's algorithm expands states in order of path cost `g`. A\* (Hart, Nilsson and
Raphael, 1968) orders them by `f = g + h`, where `h` estimates the remaining cost. If `h` never overestimates
(*admissible*), A\* returns an optimal path. If `h(u) ≤ c(u, v) + h(v)` for every edge (*consistent*), A\* also never
needs to re-expand a state.

**Bounded suboptimality.** Weighted A\* (Pohl, 1970) orders by `g + W·h`. With a consistent `h` it returns a path of
cost at most `W ×` optimal. This guarantee depends on `h` being admissible; it does not hold for a learned,
overestimating `h`.

**Focal Search** (Pearl and Kim, 1982) separates the two jobs of a heuristic. OPEN is ordered by a safe
`f = g + h_adm`. FOCAL is the subset of OPEN whose `f` is within `W ×` the current minimum. A second heuristic —
which may overestimate — decides which FOCAL state to expand next. The bound comes from the safe heuristic and the
guidance from the second one.

**Learned heuristics.** A network can learn the cost-to-go from examples. In our setting the labels are exact: one
backward Dijkstra run from a target gives the true optimal cost from every node. Conditioning the network on the
user's weight vector lets one model serve every slider setting.

**node2vec** (Grover and Leskovec, 2016) learns node embeddings from biased second-order random walks with a
skip-gram objective (Mikolov et al., 2013). Its return parameter `p` and in-out parameter `q` move the walks between
breadth-first and depth-first exploration.

### 2.2 Related work

**Personalised and pleasant pedestrian routing.** Quercia, Schifanella and Aiello (2014) route people along
beautiful, quiet or happy streets using crowd-sourced perceptions of street scenes. Novack, Wang and Zipf (2018)
build customised pleasant pedestrian routes from OpenStreetMap data, with greenery, social places and traffic noise
as factors. Delling et al. (2015) infer drivers' personal preferences from GPS traces. These systems show the demand
for preference-aware routing. They concentrate on *what* to optimise and use standard shortest-path search. We take
the cost side from open data in the same spirit, and study the *search* side: how hard such costs make A\*, and
whether learning can recover the lost speed.

**Multi-objective shortest paths.** Hansen (1980) and later Multiobjective A\* (Mandow and Pérez de la Cruz, 2010)
compute the full Pareto set of routes. That set can be very large, and a user cannot browse it interactively. We
instead scalarise the objectives with the user's slider weights, which gives one best route per setting. The price
is that the cost function, and therefore the quality of any heuristic, changes with every slider move.

**Speed-up techniques for road networks.** ALT (Goldberg and Harrelson, 2005) derives strong admissible bounds from
precomputed landmark distances. Contraction Hierarchies (Geisberger et al., 2008) and Customizable Route Planning
(Delling et al., 2017) preprocess the graph for very fast queries; CRP is designed for metrics that change, but it
needs a fresh customisation step for each new metric. Bast et al. (2016) survey the field. These methods are much
faster than ours on large graphs. We left them out of scope, as the proposal did, to isolate the effect of the
heuristic; Section 4.9 shows that a stronger admissible bound such as ALT is exactly what Focal Search would need.

**Learned and inadmissible heuristics.** Ernandes and Gori (2004) trained networks as "likely-admissible"
heuristics. Samadi, Felner and Schaeffer (2008) learned to combine several heuristics, and Arfaee, Zilles and Holte
(2011) bootstrapped learned heuristics for large state spaces. Neural A\* (Yonetani et al., 2021) learns guidance for
grid path planning end-to-end. Unlike most of this work, our labels are *exact*, and the network is conditioned on
the cost function itself, so one model serves every slider setting.

**Bounded-suboptimal search with inadmissible guides.** Thayer and Ruml (2011) observed that focal-list methods
can make poor use of an informed inadmissible estimate, and proposed Explicit Estimation Search (EES), which also
uses the inadmissible estimate to decide when to raise the lower bound. Our focal-floor measurement (Section 4.9) quantifies
the same limitation on a real street graph, and shows that it holds even for a perfect guide.

### 2.3 Where this project sits

We combine three things that the work above treats separately: a personalised cost from open geodata, a heuristic
learned across cost functions, and a bounded-suboptimal search that keeps a provable guarantee. Every claim is
checked against the exact optimum.

---

## 3. Method

### 3.1 Study area and data

#### 3.1.1 Choosing the area: the coverage audit

Before writing the cost model we audited candidate areas (`results/audit/`, and the proposal's Section 3). Around the
BITS campus, OSM mapping was too sparse: no street names or stairways. Miyapur passed most checks, but canopy
preference barely changed routes: 92 % overlap with the fastest route at weight 0.8 (`results/audit/comparison.csv`).

Jubilee Hills gave a working signal on every feature. On the final 7 × 7 km graph, a preference weight of 0.8:

- cut traffic exposure by 29 %, climb by 26 % and turns by 57 %;
- more than doubled canopy along the route;
- in each case at a mean detour of 12 % or less (`results/audit/jubilee_3500_verdict.csv`).

#### 3.1.2 The graph

| Property | Value |
|---|---|
| Centre, extent | 17.4220 N, 78.4200 E; 7 × 7 km; UTM zone 44N (EPSG:32644) |
| Nodes / directed edges | 13,246 / 34,708 (largest strongly connected component = 100 % of nodes) |
| Turn-aware states / transitions | 34,708 / 103,384 |
| Network length (both directions) | 2,075 km |
| Elevation relief | 121 m |
| Stairway edges | 60 |

Source: `data/processed/graph_summary.json`. All data is open: OpenStreetMap (ODbL), ESA WorldCover 2021
(CC BY 4.0) and Copernicus GLO-30 DEM. We used no purchased data, API keys or scraping.

#### 3.1.3 Features

Every feature is a per-metre score in [0, 1], except junction load, which is priced per turn. We label each one
honestly as measured, derived or proxy.

| Feature | Source | Status | Definition | Mean (length-weighted; junction: per transition) |
|---|---|---|---|---|
| Walking time | OSM highway class, length | measured | `0.5 / speed`; 1.3 m/s on footways … 0.5 m/s on steps | 0.42 |
| Junction load | node degree, way geometry | derived, exact | turn angle from bearings (<30° → 0, <75° → 0.3, <150° → 1, U-turn → 3), +0.2 at nodes with ≥ 3 neighbours | 1.61 |
| Traffic exposure | OSM highway class | proxy | ordinal rank: trunk/primary 1.0 … footway 0.0 | 0.39 |
| Canopy proximity | ESA WorldCover 2021, 10 m | proxy | `1 − tree-cover share` in a 15 m buffer | 0.91 |
| Climb | Copernicus GLO-30 DEM; `highway=steps` | derived, reliable | uphill grade clipped at 15 %, scaled to [0, 1]; stairs = 1 | 0.12 |

Source: `results/feature_summary.csv`.

**Why "canopy", not "shade".** Real shade modelling needs building heights and the sun's position. Building-height
tags in Hyderabad OSM are too sparse for that. What we can defensibly compute is proximity to tree cover. We call it
canopy proximity and label it a proxy: it correlates with shade but does not measure it. Likewise, OSM carries no
traffic counts, so traffic exposure is an ordinal ranking of road classes.

### 3.2 Problem formulation

#### 3.2.1 Cost model

For a weight vector `w = (w_time, w_junction, w_traffic, w_canopy, w_climb)` with every slider in [0, 1]:

```
rho(e, w)          = 0.1 + Σ_k w_k · f_k(e)            k ∈ {time, traffic, canopy, climb}
c(e_in, e_out, w)  = length(e_out) · rho(e_out, w)  +  w_junction · 30 m · turn_penalty(e_in, e_out)
```

The base term 0.1 keeps every metre's cost above zero for every weight vector. Without it, the admissible
heuristic below would collapse to zero whenever all non-distance weights are zero on some edge. A penalty-1 turn
costs as much as 30 m of walking at full junction weight.

#### 3.2.2 Turn-aware state space

A search state is the pair *(node, incoming edge)* — equivalently, the edge just walked. Transition `e → e'` exists
when `e'` leaves the node `e` enters, and it costs `c(e, e', w)`. The origin is a virtual state with no incoming
edge, so the first edge carries no turn cost. A query is solved when a state whose head is the destination is
expanded. U-turns are allowed (penalty 3) and self-loops (cul-de-sacs) are kept, so turning around is priced, not
forbidden.

#### 3.2.3 The admissible heuristic, and why it is consistent

```
h_adm(v, t, w) = rho_min(w) · s · haversine(v, t),      rho_min(w) = min_e rho(e, w)
```

OSMnx (Boeing, 2017) measured edge lengths as sums of great-circle segments on a sphere of radius 6,371,009 m. We use
haversine on the same sphere. `s` is the smallest `length(e) / haversine(u, v)` over all edges, times `(1 − 10⁻⁹)`
for rounding; on our graph `s = 0.999999997`. Then for every edge `e = (u, v)` and any incoming state:

```
h_adm(u) ≤ rho_min · s · (haversine(u, v) + haversine(v, t))    triangle inequality
         ≤ length(e) · rho(e, w) + h_adm(v)                        definition of s, rho_min
         ≤ c(e_in, e, w) + h_adm(v)                                turn penalties ≥ 0
```

Since `h_adm(t) = 0`, `h_adm` is consistent and therefore admissible. The test suite checks this inequality on every
edge of the real graph, and the label pipeline checked `h_adm ≤ h*` on all 8,500,000 labelled samples: 0 violations
(`data/labels/labels_summary.json`).

### 3.3 The learned heuristic

#### 3.3.1 Exact labels from backward Dijkstra

For a target `t` and weight vector `w`, one Dijkstra run on the *reversed* state graph gives `D(e)`: the optimal cost
from every state to `t`. The node-level label is the optimal cost of a route that *starts* at `v`:

```
H(v) = min over edges e' leaving v of [ cost(e') + D(e') ],      H(t) = 0
```

Turn penalties are non-negative, so `H(head(e)) ≤ D(e)`. A node-level heuristic trained on `H` therefore targets a
lower bound of the true state cost-to-go.

The 8,000 backward runs (40 weight vectors × 200 targets) use SciPy's compiled Dijkstra and take about one to two
minutes on one core. The test suite cross-checks them against a pure-Python reference implementation. From each run
we keep 1,000 uniformly sampled nodes; every label is exact, and sampling only bounds the dataset size.

**The published dataset** (`data/labels/`, described in `DATASHEET.md`):

| Folder | Weight vectors | Targets | Runs | Samples |
|---|---|---|---|---|
| `train_weights/` | 40 training | all 200 | 8,000 | 8,000,000 |
| `heldout_weights/` | 10 held-out | 50 (val + test) | 500 | 500,000 |

Each folder holds one compressed shard per weight vector (`w00.npz` …, about 3.4 MB each) and an `index.json` with
the weight vectors and a SHA-256 per shard, which is checked whenever the labels are loaded.

#### 3.3.2 Weight vectors and splits

The weight vectors are frozen once in `data/splits/weights.json`: 40 for training and 10 held out. Each slider is
drawn uniformly from [0, 1] and switched off with probability 0.3, to cover sparse preferences.

The 200 label targets are frozen in `data/splits/targets.json` and split **by target node**: 150 train, 25
validation, 25 test. The model therefore never sees a test destination, and never sees a held-out weight vector
while training or choosing hyperparameters.

#### 3.3.3 Inputs: node2vec embeddings

We implemented node2vec from scratch. It uses second-order biased walks (`p = 1`, `q = 0.5`; 10 walks of length 40
per node, 132,460 walks in total), vectorised over all walks at once. Training is skip-gram with 5 negative samples
(window 5) for 2 epochs, with the loss falling from 0.220 to 0.095, giving 32-dimensional embeddings
(`data/processed/node2vec.json`).

#### 3.3.4 Network and ratio parameterisation

```
x       = [ node2vec(v), node2vec(t), w, (Δx, Δy, ‖Δ‖) / 1 km ]   32 + 32 + 5 + 3 = 72 inputs
r_hat   = 1 + softplus( MLP(x) )                                 MLP: 72 → 256 → 256 → 1, ReLU
h_learn = r_hat · h_adm(v, t, w)
```

The network predicts the *ratio* of the true cost to `h_adm`, not the cost itself. Since `r_hat ≥ 1`, `h_learn` is
never less informed than the classical heuristic, and the target ratio `h*/h_adm` is scale-free across weight
vectors. The loss is the mean squared error of `log r_hat` against `log(h*/h_adm)`.

#### 3.3.5 Training

We used Adam (Kingma and Ba, 2015) with batches of 4,096, halved the learning rate when validation stalled, and
stopped early with patience 3, checkpointing the best epoch. A hyperparameter pass over hidden size × learning rate,
on 1.5 M samples for 3 epochs, chose 256 × 256 at lr = 0.003:

| Hidden | Learning rate | Validation MSE (log ratio) |
|---|---|---|
| 256, 256 | 0.003 | **0.0155** |
| 128, 128 | 0.003 | 0.0165 |
| 128, 128 | 0.001 | 0.0181 |
| 256, 256 | 0.001 | 0.0182 |

Three seeds were then trained on all 6,000,000 training-target samples. Their best validation MSEs were 0.0169,
0.0138 and 0.0161, at epochs 1, 10 and 2 (`results/training/`). The training loss keeps falling while the validation
loss stalls early, which shows that generalising to *unseen destinations* is the hard part with only 150 training
targets.

![Training curves](../../results/plots/training_curves.png)

#### 3.3.6 Batched inference with cached embeddings

The first layer is linear, so `W₁x` splits into a node term, a target term, a weight term and an offset term. The
node and target terms are precomputed once for every node (the cached embeddings), and the weight and target terms
once per query. Only the 3-number offset and the remaining layers run per node, in numpy.

When the search first meets a node, the whole 250 m spatial tile containing it is inferred in one batch, so only the
region the search actually touches is computed. A test checks that this numpy port agrees with PyTorch.

#### 3.3.7 Tuned classical fallback

The proposal's Week-4 fallback needs a strong *classical* guide. We fitted `h_tuned = κ(w) · h_adm`, with
`log κ` linear in `w`, by least squares on the training labels. It uses no node information, so it is a tuned version
of the classical heuristic rather than a learned one (`models/tuned_classical.json`).

### 3.4 Search algorithms

All searches run on the turn-aware state graph and count states **expanded** (taken off OPEN; the origin counts
as one) and **generated** (inserted into OPEN, new or with a better `g`).

**Dijkstra, A\*, weighted A\*.** One best-first search ordered by `g + W·h`, breaking ties toward larger `g`. With
the consistent `h_adm`, A\* never re-expands and weighted A\* keeps its `W ×` bound without re-expansions.

**Focal Search.** OPEN is ordered by `f = g + h_adm`. FOCAL holds OPEN states with `f ≤ W · f_min`. The next state is
the FOCAL state with the smallest `g + h_learn`. The implementation keeps three heaps with lazy deletion and per-state
version stamps. When `f_min` rises, states move from a waiting heap into FOCAL, and a FOCAL entry found above the
current bound is moved back. Re-opening is on.

**Why the bound holds.** With re-opening, some state on an optimal path is always on OPEN with its optimal `g`, so
`f_min ≤ C*`. A goal is only ever taken from FOCAL, where `g = f ≤ W · f_min ≤ W · C*`.

**Focal floor.** For every query and W we also count the states with `g* + h_adm < C*/W`, using a forward Dijkstra.
Focal Search stops only once `f_min ≥ C*/W`, and with re-opening each such state keeps a lower-`f` state on OPEN
until it is expanded. So **every** Focal Search must expand all of them, whatever its guide. The test suite checks
this lower bound for perfect and adversarial guides.

The tests also run Focal Search with a random, wildly inadmissible guide at W ∈ {1.0, 1.1, 1.5, 2.0}: the bound
always holds, and W = 1.0 is always exactly optimal.

### 3.5 The application

* **Pipeline** (`python -m wayfinder <step>`): preprocess → splits → benchmark → labels → node2vec → train →
  evaluate → report → study-prepare / study-analyze. Each step only reads and writes files. Splits and benchmark are
  frozen and refuse to be overwritten. All settings live in `configs/default.yaml`.
* **Routing service** (FastAPI): `POST /api/route` takes an origin, a destination and the five weights, snaps the
  points to the network, searches, and returns the route geometry, its metrics, search statistics and the fastest
  route for comparison. Route pricing is cached per exact weight vector, behind a lock, because requests arrive on a
  thread pool.
* **Web interface** (Leaflet): click a start and a destination, then drag five horizontal sliders or pick a preset
  (fastest, shade, quiet, easy). The route updates as you drag, and only the newest request may draw, so a slow
  earlier reply cannot overwrite it. **By default the search is exact (A\*)**: the displayed route is the best
  possible route for the chosen sliders. Focal Search and weighted A\* are opt-in modes labelled "within W × best".
  The layout also works on a phone-width screen.
* **Tests:** 87 tests, run on a synthetic fixture graph and on the real graph (Section 8).

**Handling bad and unusual input.** An application is judged on the cases a user does not plan for. Each case below
is handled with a clear message rather than a crash, and the starred ones are covered by tests:

| Situation | What happens |
|---|---|
| Point clicked outside the study area (more than 300 m from the walking network)\* | HTTP 400 with "pick a point inside the study area"; no stale route stays on the map |
| Start and destination snap to the same node | message "Start and destination are at the same place. Move one of the markers." |
| W below 1\* | rejected by request validation (HTTP 422) |
| Slider value outside [0, 1] (possible only through the API) | rejected by request validation (HTTP 422) |
| Unknown search algorithm\* | HTTP 400 listing the valid ones |
| All sliders at zero | still well defined: the base term 0.1 makes the cost proportional to length, so the shortest route is returned |
| Fast slider dragging, many requests in flight\* | answers stay correct under concurrent requests; only the newest answer is drawn |
| Study page with no trials prepared\* | HTTP 404 telling the operator which command to run |
| Pipeline step run in the wrong order or on missing files\* | one-line `error: …` message and exit code 1, no traceback |

---

## 4. Results and evaluation

### 4.1 Experimental setup

* **Benchmark:** 200 queries frozen with a fixed seed (`data/benchmark/queries.csv`, with a SHA-256 checked on every
  load). Origin and destination are at least 700 m apart (median 3.4 km, max 7.7 km), and no destination is a label
  target. 100 queries use a training weight vector and 100 a held-out one.
* **Ground truth:** the exact optimum of every query, from backward Dijkstra.
* **Conditions:**
    - Dijkstra; A\*(h_adm); weighted A\* at W = 1.25, 1.5, 2;
    - A\*(h\*) (oracle);
    - A\*(h_learn) (no bound);
    - Focal Search guided by h_learn, by h_tuned and by h\* (oracle), each at W = 1.0, 1.1, 1.25, 1.5, 1.75, 2.0.
    - Learned conditions are run with all three seeds (7,800 runs in total).
* **Metrics:**
    - states expanded and generated;
    - latency (graph pricing + heuristic preparation + search, one process);
    - suboptimality (cost / exact optimum) and bound violations;
    - heuristic admissibility violations, consistency violations and informedness `h/h*`.
* **Controls:** the perfect heuristic `h*` is run as an oracle, both in A\* and as the Focal guide. It shows the
  best any guide could do, so a weak result can be traced to the guide or to the search framework.
* **Hardware:** a laptop CPU, single-threaded Python, no GPU.

### 4.2 The problem, stated as a measurement (H1)

The λ family `w = (1 − λ, λ/4, λ/4, λ/4, λ/4)` moves weight from walking time to the four other sliders. This is
on the 200 frozen origin–destination pairs (`results/problem_curve.csv`):

| λ | rho_min | Dijkstra | A\*(h_adm) | A\*(h_adm) / Dijkstra | h_adm / h\* at origin | A\*(h\*) |
|---|---|---|---|---|---|---|
| 0.0 | 0.485 | 17,316 | 6,026 | **0.35** | 0.67 | 87 |
| 0.2 | 0.408 | 17,321 | 7,787 | 0.45 | 0.56 | 85 |
| 0.4 | 0.331 | 17,324 | 9,617 | 0.56 | 0.46 | 84 |
| 0.6 | 0.254 | 17,311 | 11,404 | 0.66 | 0.35 | 84 |
| 0.8 | 0.177 | 17,286 | 13,105 | 0.76 | 0.25 | 85 |
| 0.9 | 0.138 | 17,272 | 13,974 | **0.81** | 0.19 | 85 |

(mean states expanded per query)

![Problem curve](../../results/plots/problem_curve.png)

As preferences move away from distance, `rho_min` falls, `h_adm` recovers less and less of the true cost (67 % → 19 %),
and A\* drifts toward Dijkstra. The perfect heuristic would need about 85 expansions throughout, so the headroom a
better heuristic could in principle exploit grows from about 70× at λ = 0 to about 160× at λ = 0.9. The figure also
shows learned Focal Search at W = 1.5: at low λ it expands *more* than A\*, and it never comes close to the
headroom. Section 4.9 explains why.

### 4.3 Search efficiency on the frozen benchmark

Medians over the 200 queries; learned conditions pool three seeds (`results/benchmark_summary.csv`):

| Condition | W | Expanded (median) | IQR | Focal floor | Latency median / p95 (ms) | Mean / max suboptimality |
|---|---|---|---|---|---|---|
| Dijkstra | – | 17,574 | 6,975 – 26,970 | – | 72 / 178 | 1 / 1 |
| A\*(h_adm) | 1 | 13,168 | 5,398 – 22,265 | – | 63 / 183 | 1 / 1 |
| Weighted A\* | 1.5 | 11,054 | 4,847 – 20,154 | – | 56 / 184 | 1.000 / 1.060 |
| Weighted A\* | 2 | 9,586 | 3,701 – 17,817 | – | 51 / 176 | 1.002 / 1.062 |
| Focal, tuned classical | 1.5 | 11,029 | 4,616 – 19,134 | 4,877 | 109 / 379 | 1.000 / 1.030 |
| Focal, tuned classical | 2 | 8,812 | 3,749 – 16,921 | 2,110 | 90 / 395 | 1.001 / 1.053 |
| **Focal, learned** | 1.5 | 11,161 | 4,619 – 19,252 | 4,877 | 261 / 783 | 1.001 / 1.056 |
| **Focal, learned** | 2 | 8,552 | 3,555 – 17,619 | 2,110 | 204 / 726 | 1.002 / 1.108 |
| Focal, perfect guide h\* (oracle) | 1.5 | 10,847 | 4,270 – 18,260 | 4,877 | 110 / 340 | 1 / 1 |
| Focal, perfect guide h\* (oracle) | 2 | 8,049 | 2,820 – 15,530 | 2,110 | 76 / 304 | 1 / 1 |
| **A\*(h_learn), no bound** | – | **1,205** | 402 – 2,351 | – | **40** / 144 | 1.044 / 1.506 |
| A\*(h\*) (oracle) | 1 | 84 | 54 – 112 | – | 6 / 11 | 1 / 1 |

Per-seed medians for learned Focal at W = 1.5 are 11,285.5, 11,060 and 11,269, and for A\*(h_learn) 1,340,
1,031.5 and 1,177.5. The seeds agree closely, so the differences between conditions are not seed noise. The IQRs
are wide because query length varies from 0.7 to 7.7 km; all conditions are compared on the same 200 queries.

![Expansions against W](../../results/plots/expansions_vs_W.png)

### 4.4 The suboptimality bound holds

Across all 7,000 bounded runs (every Focal condition, every W, three seeds, plus weighted A\* and A\*), **no route
exceeded its guaranteed bound** (`results/bound_sweep.csv`). In practice the routes are far better than guaranteed.
For learned Focal, the worst query was 1.006×, 1.033×, 1.056×, 1.056× and 1.108× the optimum at
W = 1.1, 1.25, 1.5, 1.75 and 2.0, and the mean stayed within 0.2 % of optimal.

![Guaranteed vs achieved](../../results/plots/bound_guaranteed_vs_achieved.png)

### 4.5 Heuristic quality (H2)

Averaged over the 200 benchmark queries, at 2,000 sampled nodes each (`results/heuristic_quality.csv`):

| Heuristic | h / h\* mean (median) | Overestimates h\* | Mean overestimate | Consistency violations |
|---|---|---|---|---|
| h_adm | 0.167 (0.168) | 0 % | 0 % | 0 % |
| h_tuned | 1.029 (1.038) | 50.2 % | 13.3 % | 24.3 % |
| h_learn, seed 0 | 1.024 (1.014) | 53.5 % | 10.7 % | 32.7 % |
| h_learn, seed 1 | 1.031 (1.019) | 55.5 % | 10.4 % | 31.8 % |
| h_learn, seed 2 | 1.022 (1.009) | 51.2 % | 10.6 % | 33.2 % |

![Informedness](../../results/plots/informedness_hist.png)

The learned heuristic recovers essentially the whole true cost: its median `h/h*` is within 2 % of 1. It
overestimates about half the time, by about 10 % on average, which is why it needs Focal Search's safety net. Its
spread is tighter than the tuned classical heuristic's. The tuned heuristic is surprisingly good *on average*,
because one scalar κ(w) per weight vector already captures the typical detour ratio, but it is wider: its histogram
has a long right tail.

### 4.6 Generalisation to held-out preferences (H2)

On label level, for unseen test targets (median over seeds; `results/training/label_metrics.csv`):

| | MAE of log(h\*/h_adm) | h_learn / h\* | h_adm / h\* |
|---|---|---|---|
| training weight vectors | 0.100 | 1.041 | 0.211 |
| **held-out weight vectors** | 0.104 | 1.013 | 0.117 |

On search level, the median expanded states split by weight set (`results/generalisation.csv`): A\*(h_learn) needs
1,068 on held-out weights against 1,340 on training weights. Learned Focal at W = 1.5 needs 11,726 against 11,065.

The network **generalises rather than memorises**: its accuracy on preferences it never saw matches the training
preferences. The held-out queries are harder for the classical heuristic, whose informedness drops to 0.117, but not
for the learned one.

![Generalisation](../../results/plots/generalisation.png)

### 4.7 Latency

All conditions run in single-threaded Python and are interactive (`results/plots/latency.png`). A\*(h_adm) has a
median of 63 ms. A\*(h_learn) is the fastest practical option at 40 ms, because it expands 11× fewer states. Learned
Focal is the slowest, at 261 ms median and 783 ms p95: it pays for network inference *and* expands as many states as
A\*. Absolute times depend on the machine; on the reproduction run (Section 4.11) every condition was about 30 %
faster, with the same order.

![Latency](../../results/plots/latency.png)

### 4.8 Week-4 go / no-go (H3)

The proposal's gate asks whether the learned heuristic expands fewer nodes than the classical baseline. We applied it
strictly: GO requires beating both A\*(h_adm) and the tuned classical guide at the same W
(`results/go_no_go.json`).

* Learned Focal against A\*(h_adm) at W = 1.5: 11,161 vs 13,168, **15.2 % fewer** ✔
* Learned Focal against tuned Focal at W = 1.5: 11,161 vs 11,029 ✘
* Learned Focal against weighted A\* at W = 1.5: 11,161 vs 11,054 ✘

**Verdict: NO-GO.** Following the proposal's fallback, the headline method is Focal Search guided by the tuned
classical heuristic, and the rest of this section is the measurement of why the learned version did not pay off.

### 4.9 Why Focal Search cannot exploit a good heuristic

The learned heuristic is excellent (Section 4.5), and A\* with it is 11× cheaper. Yet inside Focal Search it barely
helps — and **neither does a perfect heuristic**. Focal Search guided by the true `h*` saves only 17.6 % at W = 1.5.

The reason is structural. A goal can only be taken from FOCAL once `f_min ≥ C*/W`, and `f_min` is the minimum of
`g + h_adm` over OPEN. With `h_adm` recovering only 17 % of the true cost, the region of states with
`g + h_adm < C*/W` is large, and every one of those states must be expanded before the search can stop.

At W = 1.5 this **focal floor** is 4,877 states (median). The observed expansions are above 10,800 for *every*
guide, because the guide spends expansions on near-optimal states while `f_min` still has to be raised.

Two practical consequences follow:

- The guarantee of Focal Search is only as useful as its admissible heuristic. Better guidance cannot buy speed
  until `h_adm` itself is stronger, for example with ALT landmarks (Goldberg and Harrelson, 2005), which the
  proposal placed out of scope.
- Where a guarantee is not required, the learned heuristic in plain A\* gives most of the possible gain. It cuts
  expansions 11×, keeps routes within 4.4 % of optimal on average, and reaches 51 % worse only in the worst query.

We also piloted a bound-preserving "incumbent" FOCAL policy, which switches to the `f_min` state once a goal has been
generated. On eight benchmark queries it saved at most about 3 %, so we removed it: the extra expansions happen
*before* the goal is ever generated.

### 4.10 The application returns the best route

`scripts/verify_routes.py` drew 147 random trips with random slider settings on the real graph. Every route the
application returned by default equalled the exact Dijkstra optimum, both by its reported cost and when re-priced
from its edges (**147/147**, worst ratio 1.000000000000).

For an exact optimiser, raising one slider cannot increase the route's total of what that slider penalises. We
checked this for all five sliders on every trip: **0 violations in 735 checks**. Moving each slider from 0 to 1
changed the route in 106 (time), 141 (junction), 136 (traffic), 129 (canopy) and 115 (climb) of the 147 trips. So
every slider has a visible effect for most trips.

### 4.11 Reproducibility from a fresh clone

To check that the repository runs as submitted, we cloned it into an empty folder, installed it into a new virtual
environment and ran the test suite, `scripts/verify_routes.py` and the whole pipeline (`python -m wayfinder all`).

* `pip check` reported no conflicts, all 87 tests passed, and route verification gave 147/147.
* The full pipeline finished in 45 minutes with exit code 0.
* The processed graph, the 8,500,000 labels, the node2vec embeddings and all three trained models were rebuilt
  **byte-identical** to the committed files.
* All 7,800 benchmark runs, 4,800 problem-curve runs and the generalisation table matched the committed results in
  every column except the timing columns. The go / no-go verdict and every expansion count in this section were the
  same.

---

## 5. Discussion and limitations

### 5.1 The hypotheses revisited

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: the admissible heuristic weakens with non-distance weight | **supported** | A\* / Dijkstra expansions rise from 0.35 to 0.81 as λ goes from 0 to 0.9 (Section 4.2) |
| H2: the cost-to-go is learnable across preferences | **supported** | mean `h_learn / h*` 1.02 against 0.17 for `h_adm`; same accuracy on held-out weight vectors (Sections 4.5–4.6) |
| H3: learned guidance speeds up bounded search | **not supported** | 15 % fewer expansions than A\*, but no better than weighted A\* or the tuned guide at the same W; a perfect guide gains only 18 % (Sections 4.8–4.9) |

**What worked.**

- The cost model gives a real signal on every slider.
- The admissible heuristic is provably consistent.
- Exact labels make learning easy to evaluate.
- The preference-conditioned network is accurate and generalises to unseen preferences.
- Focal Search's bound held in every one of 7,000 runs.
- The application returns provably best routes interactively.

**What did not.** The proposal's headline — learned guidance making bounded-suboptimal search substantially faster —
did not materialise. The cause is not the network but the search framework: Focal Search's effort is set by the
admissible OPEN heuristic. We consider the focal-floor measurement, together with the perfect-guide control, the main
technical finding of the project.

### 5.2 Impact

**Who it is for.** Pedestrians whose idea of a good route is not the fastest one: people avoiding stairs and
gradients, people avoiding busy roads, and people looking for tree cover in the heat. The application lets them state
that trade-off directly with five sliders, and see the cost of it against the fastest route.

**What a user gets.** For any slider setting, the exact best route for the stated weights, in tens of milliseconds
on a laptop, with its length, time and each objective's total next to the fastest route. On 147 random trips each
slider changed the route in 72–96 % of cases (Section 4.10), so the controls are not cosmetic.

**What the search finding means in practice.** For interactive routing on a city-district graph, exact A\* is already
fast enough, and it is the right default. If a much larger area made A\* too slow, the learned heuristic in plain A\*
is the useful speed-up (11× fewer expansions, 4.4 % mean detour from optimal). Focal Search would only become
worthwhile with a stronger admissible bound.

### 5.3 Limitations and threats to validity

* *Proxies.* Traffic exposure is ranked from road class, not measured. Canopy is tree-cover proximity, not shade.
  The climb score inherits 30 m DEM noise; grades are clipped at 15 % for that reason.
* *One study area.* All results are on one 7 × 7 km graph. The coverage audit shows that sparse OSM areas such as
  the BITS campus would behave differently. A new area needs a new graph and new labels; the model does not
  transfer between graphs.
* *No user evaluation.* We verified that every route is optimal for the cost function we wrote down, not that people
  agree with that cost function. A 12-trial study page with counterbalanced route order is implemented, but it was
  not run, and we report no agreement rate.
* *Generalisation to destinations.* With 150 training targets, validation loss stalls after a few epochs. More
  targets would likely tighten the learned heuristic further.
* *Latency* is measured in single-threaded Python on one laptop. The relative order of conditions is what matters;
  absolute times would fall several-fold in a compiled implementation.
* *Static data.* The graph is a snapshot: there is no live traffic, no time-of-day shade and no closures.
* *Local deployment.* The service runs on the user's machine (`python -m wayfinder serve`); it is not hosted online.

### 5.4 Future work

- A stronger admissible bound for OPEN, such as ALT landmarks or the max of several admissible estimates, so that
  Focal Search can use its guide.
- Explicit Estimation Search, which uses an inadmissible estimate to decide *when* to raise `f_min`.
- Training against an admissible ceiling, to trade a little informedness for fewer overestimates.
- Running the user study, even at small scale, and reporting agreement with its participant count.

---

## 6. Individual contributions

### 6.1 Who did what

The team worked in four pairs, each owning one part of the system end to end. The interfaces between parts (array
formats, file layouts, function signatures) were fixed in Week 1, so the pairs could work in parallel
(`docs/IMPLEMENTATION_PLAN.md`).

| Pair | Members | Build phase (Weeks 1–4) | Second phase (Weeks 5–7) | Main code |
|---|---|---|---|---|
| Graph | Anisha Potdar, Ananya Avunooru | OSM extraction, coverage audit, UTM projection, turn-aware graph, five feature extractors, cost model | routing service (FastAPI), web interface with sliders | `scripts/acquisition/`, `src/wayfinder/graph/`, `geo.py`, `cost.py`, `service/`, `web/` |
| Search | Srujana N S, Anushka Sahoo | Dijkstra, A\*, weighted A\*, admissible heuristic, turn-aware expansion, node accounting, batched inference | Focal Search, W sweep, empirical bound verification | `src/wayfinder/search/`, `learning/inference.py`, `tests/test_search.py` |
| Data | Soujanya Lakshmi Tayyala, Chandana N | backward Dijkstra at scale, weight-vector sampling, label pipeline, dataset publication, node2vec | evaluation harness, heuristic-quality analysis, plots | `search/backward.py`, `learning/labels.py`, `weights.py`, `splits.py`, `node2vec.py`, `evaluation/` |
| Model | Vinny Polamarasetty, Chaitanya Srikar Kastury | network architecture, ratio parameterisation, training loop, hyperparameter pass, checkpointing | generalisation on held-out weights, user study tooling | `learning/model.py`, `train.py`, `artifacts.py`, `study/`, `web/study.*` |

Each pair is responsible for explaining its components in the viva.

### 6.2 Work over time

The work followed the two phases of the proposal's schedule:

| Period | Work |
|---|---|
| Week 1 | Study-area coverage audit; Jubilee Hills chosen over the BITS campus and Miyapur. The dataset, build scripts and audit were merged into the shared repository through a pull request on 2 October 2026. Interfaces frozen. |
| Weeks 1–4 (build phase) | Graph, features and cost model; Dijkstra / A\* / weighted A\*; label pipeline and node2vec; network and training. |
| Week-4 checkpoint | Go / no-go gate on the frozen benchmark: NO-GO, fallback to the tuned classical guide (Section 4.8). |
| Weeks 5–7 (second phase) | Focal Search and the bound sweep; evaluation harness, focal-floor analysis and plots; routing service, web interface and study page. |
| Before submission | Correctness audit against exact optima, fresh-clone reproduction run, report. |

### 6.3 Honest reflection

* **The negative result could have been predicted earlier.** The perfect-guide control (Focal Search guided by `h*`)
  needs no training at all. Had we run it first, it would have shown before any training that no guide could beat the tuned
  classical heuristic by much inside Focal Search. We would have spent the time on a stronger admissible bound (ALT)
  instead.
* **Exact ground truth paid for itself.** Because every route can be checked against backward Dijkstra, our audit
  found real bugs that ordinary testing would have missed. The web app first defaulted to Focal Search at W = 1.5,
  so it did not always show the best route. Its cache rounded weights, so two nearby slider settings could share a
  route. The focal-floor count included goal states through floating-point ties. All three were fixed.
* **The proposal over-scoped the user study.** Recruiting 25 participants was not feasible in the time available, so
  the study was not run. We did not simulate or invent responses.
* **Our commit history under-represents the timeline.** Most of the implementation reached the shared repository in
  a few large commits in November, instead of one pull request per feature from the start. Next time each pair would
  work on its own branch and merge through reviewed pull requests, as we did for the Week-1 dataset.
* **Heavy AI assistance shifted the work toward verification.** Section 7 describes how the assistant was used.
  The lesson was that generated code is only as trustworthy as the checks around it, so most of our effort went into
  tests against exact optima and into reading the code well enough to explain it.

---

## 7. AI usage

**Tool.** Claude Code (Anthropic), an AI coding assistant, used through the project's implementation and write-up
phase. Its use is logged in `docs/ai_usage_log.md`.

**What it was used for.**

* Implementing and refactoring code: the pipeline, search algorithms, training and evaluation code, the FastAPI
  service, the web interface and the study tooling.
* Writing tests and the route-verification script.
* Running the pipeline and evaluation, and packaging the project (requirements, Makefile, README).
* Drafting and editing this report.

**What stayed with the team.** The project idea, the proposal (problem, objectives, algorithms, evaluation plan and
the Week-4 gate), the Week-1 data acquisition and coverage audit, and the decision of what to report.

**How we verified the output.** Nothing generated was accepted on trust:

* every search algorithm is tested against exact optima from backward Dijkstra, on a fixture graph and on the real
  graph;
* Focal Search is tested with an adversarial, wildly inadmissible guide, and its bound held in all 7,000 benchmark
  runs;
* `h_adm` admissibility and consistency are checked on every edge and on all 8,500,000 labels;
* the features, turn penalties and costs were cross-checked against an independent re-implementation from the raw
  GraphML, and the numpy inference against PyTorch;
* every number in Section 4 is read from a result file, and the full pipeline was re-run from a fresh clone
  (Section 4.11).

**Errors the assistant made.** The three bugs listed in Section 6.3 were in assistant-written code and were found
by these checks. This is why we treat the checks, not the generated code, as the evidence.

---

## 8. Reproducibility

Everything needed is in the repository. That includes the graph, frozen splits and benchmark, the label dataset,
embeddings, trained models and every result file.

```bash
python -m pip install -r requirements-dev.txt && python -m pip install -e . --no-deps
python -m pytest                       # 87 tests: fixture graph + real graph
python scripts/verify_routes.py        # 147 random trips: exact routes, slider monotonicity
python -m wayfinder serve              # web app at http://127.0.0.1:8000
python -m wayfinder all                # whole pipeline from the GraphML (45–90 min on a laptop CPU)
```

| Step | Time |
|---|---|
| Labels (8,000 runs) | 1–2 min |
| node2vec | 3–4 min |
| Training (hyperparameter pass + 3 seeds) | 15–40 min |
| Evaluation | 25–50 min |

Every number in Section 4, except the reproduction run in Section 4.11, is copied from a file under `results/`.
`python -m wayfinder report` regenerates `results/summary.md` and all figures from the raw result files.

---

## References

1. Arfaee, S. J., Zilles, S., Holte, R. C. (2011). Learning Heuristic Functions for Large State Spaces.
   *Artificial Intelligence*, 175(16–17), 2075–2098.
2. Bast, H., Delling, D., Goldberg, A., Müller-Hannemann, M., Pajor, T., Sanders, P., Wagner, D., Werneck, R. F.
   (2016). Route Planning in Transportation Networks. In *Algorithm Engineering*, LNCS 9220, 19–80.
3. Boeing, G. (2017). OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street
   Networks. *Computers, Environment and Urban Systems*, 65, 126–139.
4. Delling, D., Goldberg, A. V., Goldszmidt, M., Krumm, J., Talwar, K., Werneck, R. F. (2015). Navigation Made
   Personal: Inferring Driving Preferences from GPS Traces. *ACM SIGSPATIAL 2015*.
5. Delling, D., Goldberg, A. V., Pajor, T., Werneck, R. F. (2017). Customizable Route Planning in Road Networks.
   *Transportation Science*, 51(2), 566–591.
6. Ernandes, M., Gori, M. (2004). Likely-Admissible and Sub-Symbolic Heuristics. *ECAI 2004*.
7. Geisberger, R., Sanders, P., Schultes, D., Delling, D. (2008). Contraction Hierarchies: Faster and Simpler
   Hierarchical Routing in Road Networks. *WEA 2008*, LNCS 5038, 319–333.
8. Goldberg, A. V., Harrelson, C. (2005). Computing the Shortest Path: A\* Search Meets Graph Theory. *SODA 2005*.
9. Grover, A., Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. *KDD 2016*.
10. Hansen, P. (1980). Bicriterion Path Problems. In *Multiple Criteria Decision Making: Theory and Application*,
    Lecture Notes in Economics and Mathematical Systems 177, Springer, 109–127.
11. Hart, P. E., Nilsson, N. J., Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost
    Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.
12. Kingma, D. P., Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
13. Mandow, L., Pérez de la Cruz, J. L. (2010). Multiobjective A\* Search with Consistent Heuristics. *Journal of the
    ACM*, 57(5).
14. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). Distributed Representations of Words and
    Phrases and their Compositionality. *NeurIPS 2013*.
15. Novack, T., Wang, Z., Zipf, A. (2018). A System for Generating Customized Pleasant Pedestrian Routes Based on
    OpenStreetMap Data. *Sensors*, 18(11), 3794.
16. Pearl, J., Kim, J. H. (1982). Studies in Semi-Admissible Heuristics. *IEEE TPAMI*, 4(4), 392–399.
17. Pohl, I. (1970). Heuristic Search Viewed as Path Finding in a Graph. *Artificial Intelligence*, 1(3–4), 193–204.
18. Quercia, D., Schifanella, R., Aiello, L. M. (2014). The Shortest Path to Happiness: Recommending Beautiful,
    Quiet, and Happy Routes in the City. *ACM Hypertext 2014*.
19. Russell, S., Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*, 4th ed., Chapters 3–4 and 22.
20. Samadi, M., Felner, A., Schaeffer, J. (2008). Learning from Multiple Heuristics. *AAAI 2008*.
21. Thayer, J. T., Ruml, W. (2011). Bounded Suboptimal Search: A Direct Approach Using Inadmissible Estimates.
    *IJCAI 2011*.
22. Yonetani, R., Taniai, T., Barekatain, M., Nishimura, M., Kanezaki, A. (2021). Path Planning using Neural A\*
    Search. *ICML 2021*.

Data: OpenStreetMap contributors (ODbL); ESA WorldCover 10 m 2021 v200 (CC BY 4.0); Copernicus GLO-30 Digital
Elevation Model (European Space Agency).

---

## Appendix A — Main settings (`configs/default.yaml`)

| Setting | Value |
|---|---|
| Weight vectors | 40 training + 10 held-out; slider off with probability 0.3; seed 7 |
| Label targets | 200 (150 / 25 / 25 by target); 1,000 nodes kept per run |
| Benchmark | 200 queries, ≥ 700 m apart, half on held-out weights; seed 2025 |
| node2vec | 32-d, p = 1, q = 0.5, 10 walks × 40 steps, window 5, 5 negatives, 2 epochs |
| Network | 72 → 256 → 256 → 1, ReLU, `r_hat = 1 + softplus` |
| Training | Adam, lr 0.003, batch 4,096, ≤ 12 epochs, patience 3, seeds 0, 1, 2 |
| Evaluation | Focal W ∈ {1.0, 1.1, 1.25, 1.5, 1.75, 2.0}; weighted A\* W ∈ {1.25, 1.5, 2}; FOCAL key `g + h_learn`; headline W = 1.5 |
