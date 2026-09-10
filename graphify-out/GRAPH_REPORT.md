# Graph Report - .  (2026-07-08)

## Corpus Check
- Large corpus: 190 files · ~1,031,107 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 906 nodes · 1819 edges · 54 communities (48 shown, 6 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 116 edges (avg confidence: 0.76)
- Token cost: 270,737 input · 0 output

## Community Hubs (Navigation)
- M46 Probe Placer
- Constructive Placer (C++)
- Dataset Validation & Cost Utils
- Reconstruction Probes
- Contest Scoring & Metrics
- Legacy SA Optimizer (C++)
- GNN Training Pipeline
- RF Score Projection Model
- Solution Evaluation (HPWL/Area)
- M46 Timing Accumulator
- Sequence-pair Packer Probe
- B*-tree SA Template
- Proxy Analysis & Debug Tools
- Compaction Debug
- M49 REFINE Probe
- Legacy SA Wrapper (GNN)
- Portfolio Wrapper
- Violation Debug Tools
- Baseline Optimizers & Submission
- Contest Evaluator Harness
- Sweep & Oracle Probes
- Contest Problem & Constraints
- Placer Design Concepts
- RuntimeFactor Lever (RF axis)
- Block Data Model I
- Block Data Model II
- Input Serialization & Probes
- M46 Stage Probe
- Block Data Model III
- Profile Audit (M25)
- Reconstruction Paradigm & Ceilings
- HPWL Push Debug
- FloorSet & Intel Datasets
- Skyline Packing
- Portfolio Ceiling Probe
- M48 Cold-start Dry-run
- Fast Proxy Metrics (M47)
- Portfolio Proxy Concepts
- Baseline Generation
- Tree Builder Probe
- Differentiable Training Loss
- M47 Proxy Equivalence Gate
- Pool Index Selection
- Portfolio Breakdown Tool
- matplotlib (lib)
- numpy (lib)
- requests (lib)
- tqdm (lib)

## God Nodes (most connected - your core abstractions)
1. `_serialize_input()` - 30 edges
2. `evaluate_solution()` - 28 edges
3. `_parse_output()` - 28 edges
4. `M46Acc` - 27 edges
5. `build_opt_target_pos()` - 27 edges
6. `XYWH` - 26 edges
7. `XYWH` - 26 edges
8. `solve()` - 22 edges
9. `Item` - 20 edges
10. `solve()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `evaluate_solution()`  [INFERRED]
  reconstruct_probe.py → iccad2026contest/iccad2026_evaluate.py
- `RuntimeFactor Lever (RF axis)` --conceptually_related_to--> `Runtime Normalization`  [INFERRED]
  CLAUDE.md → iccad2026contest/README.md
- `iccad2026_evaluate.py` --references--> `shapely`  [INFERRED]
  iccad2026contest/README.md → requirements.txt
- `iccad2026_evaluate.py` --references--> `torch (PyTorch)`  [INFERRED]
  iccad2026contest/README.md → requirements.txt
- `Intel Test Suite (Lite, optimal metrics)` --semantically_similar_to--> `Intel Test Suite (Prime, optimal metrics)`  [INFERRED] [semantically similar]
  intel_testsuite_lite.md → intel_testsuite.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **RuntimeFactor Lever - Six Shipped Shots** — claude_runtimefactor, claude_m41_runtime_factor, claude_m42_big_redundant, claude_m45_band_tiers, claude_m46_hotpath_exact, claude_m47_wrapper_overhead, claude_m49_refine_band_cut [EXTRACTED 0.90]
- **Constructive Placer 5-stage Pipeline** — claude_boundary_aspect_dims, claude_mib_shape_unification, claude_cluster_construction, claude_greedy_packing, claude_compaction_lever [EXTRACTED 0.85]
- **GNN ML Route (v1->v2->v3, parked)** — gnn_training_floorplannet_v1, gnn_training_floorplannet_v2, gnn_training_floorplannet_v3, gnn_training_ill_posed, gnn_training_oracle_perm [INFERRED 0.80]
- **Cost Function Components (Eq. 2)** — iccad2026contest_floorplanningcontest_iccad_2026_v10_cost_function, iccad2026contest_floorplanningcontest_iccad_2026_v10_hpwl_gap, iccad2026contest_floorplanningcontest_iccad_2026_v10_area_gap, iccad2026contest_floorplanningcontest_iccad_2026_v10_violations_relative, iccad2026contest_floorplanningcontest_iccad_2026_v10_runtime_factor, iccad2026contest_floorplanningcontest_iccad_2026_v10_infeasible_penalty [EXTRACTED 1.00]
- **Soft Constraints (penalized, not disqualifying)** — iccad2026contest_floorplanningcontest_iccad_2026_v10_grouping_constraint, iccad2026contest_floorplanningcontest_iccad_2026_v10_mib_constraint, iccad2026contest_floorplanningcontest_iccad_2026_v10_boundary_constraint [EXTRACTED 1.00]
- **Hard Constraints (violation = infeasible, M=10)** — iccad2026contest_floorplanningcontest_iccad_2026_v10_area_target_constraint, iccad2026contest_floorplanningcontest_iccad_2026_v10_overlap_free_constraint, iccad2026contest_floorplanningcontest_iccad_2026_v10_fixed_shape_immutability, iccad2026contest_floorplanningcontest_iccad_2026_v10_preplaced_immutability [EXTRACTED 1.00]

## Communities (54 total, 6 thin omitted)

### Community 0 - "M46 Probe Placer"
Cohesion: 0.06
Nodes (79): adjacent_candidates_for_block(), Anchor, w, x, y, AnchoredCluster, movable, preplaced (+71 more)

### Community 1 - "Constructive Placer (C++)"
Cohesion: 0.06
Nodes (77): adjacent_candidates_for_block(), Anchor, w, x, y, AnchoredCluster, movable, preplaced (+69 more)

### Community 2 - "Dataset Validation & Cost Utils"
Cohesion: 0.06
Nodes (57): calculate_weighted_b2b_wirelength(), calculate_weighted_p2b_wirelength(), estimate_cost(), Tensor, Calculate the weighted Half-Perimeter Wire Length (HPWL) for block-to-block edge, Calculate weighted Half-Perimeter Wire Length (HPWL) for pin-to-block edges., Estimate the cost of a layout by evaluating area and wire length violations., DataLoader (+49 more)

### Community 3 - "Reconstruction Probes"
Cohesion: 0.08
Nodes (48): bbox(), _bisect_area(), _bisect_index(), _build_tree(), main(), OFFLINE ONLY — index/attribute leak reconstruction probe (never shipped).  M40 (, Split by index median (low half / high half), preserving index order., Area-balanced greedy split (largest-first), like M40's no-edge fallback. (+40 more)

### Community 4 - "Contest Scoring & Metrics"
Cohesion: 0.05
Nodes (45): The FloorSet Challenge: Data-Driven SoC Floorplanning (ICCAD 2026 Contest v10), Bounding-Box Area Gap (Areagap), Area Target / Dimensionality Constraint (Hard), Baseline Values (HPWLbaseline, Areabaseline), Boundary Constraint (Soft), Boundary Violation (Vboundary), Bounding-Box Area, Contest Repository (IntelLabs/FloorSet) (+37 more)

### Community 5 - "Legacy SA Optimizer (C++)"
Cohesion: 0.14
Nodes (34): Clock, map, boundary_snap(), calc_bbox_area(), calc_boundary_dist(), calc_hpwl_b2b(), calc_hpwl_p2b(), calc_violation() (+26 more)

### Community 6 - "GNN Training Pipeline"
Cohesion: 0.09
Nodes (22): explore_training_data(), get_training_dataloader(), Get a DataLoader for the FloorSet-Lite training data (1M samples).          Ar, Explore training data statistics.          Args:         data_path: Path to F, aspect_loss(), FloorplanNetV3, main(), pairwise_ranking_loss() (+14 more)

### Community 7 - "RF Score Projection Model"
Cohesion: 0.11
Nodes (28): band_cases(), cost(), local_total(), m42cap(), m45_fn(), m46_pool(), RF-aware projected-score model (OFFLINE, never shipped).  The official Cost mult, Deployed _RH=1.4 proxy selector (wrapper parity). (+20 more)

### Community 8 - "Solution Evaluation (HPWL/Area)"
Cohesion: 0.12
Nodes (27): calculate_bbox_area(), calculate_hpwl_b2b(), calculate_hpwl_p2b(), check_area_tolerance(), check_dimension_hard_constraints(), check_overlap(), compute_training_loss(), compute_training_loss_batch() (+19 more)

### Community 9 - "M46 Timing Accumulator"
Cohesion: 0.07
Nodes (27): M46Acc, anch, anch_members, build, cand, cands, compact, feas (+19 more)

### Community 10 - "Sequence-pair Packer Probe"
Cohesion: 0.13
Nodes (21): _bbox(), build_modules(), case_data(), compute_nsoft(), describe(), est_cost(), _FenMax, load_cases() (+13 more)

### Community 11 - "B*-tree SA Template"
Cohesion: 0.12
Nodes (10): BStarTree, MyOptimizer, Tensor, Compute (x, y, w, h) from tree structure.                  Uses proper contour, Swap width/height (90° rotation, preserves area)., Swap two blocks' dimensions., Delete and reinsert block at random position., B*-tree Simulated Annealing baseline.          REPLACE THIS CLASS WITH YOUR AL (+2 more)

### Community 12 - "Proxy Analysis & Debug Tools"
Cohesion: 0.14
Nodes (14): Per-case violation breakdown for constructive.exe. Re-runs each case through eva, Decompose area_gap for constructive.exe (base profile). For each case compare ou, OFFLINE (never shipped): verify the python_sa_solve FALLBACK is feasible on all, M37 liveness check: how many of the 100 cases have a RESHAPEABLE MIB group (no-m, compute_cost(), Compute the official contest cost.      Cost = (1 + α·(HPWL_gap + Area_gap)) ×, Run a candidate env-profile across all 100 cases; compare each case's TRUE cost, run() (+6 more)

### Community 13 - "Compaction Debug"
Cohesion: 0.19
Nodes (20): _bbox(), boundary_nudge(), compact(), count_bv(), count_gf(), csc(), layout_score(), _nsoft() (+12 more)

### Community 14 - "M49 REFINE Probe"
Cohesion: 0.24
Nodes (16): band_dt(), cost_of(), mode_trace(), mode_variant(), pm_cache(), pm_of(), pool_at(), project() (+8 more)

### Community 15 - "Legacy SA Wrapper (GNN)"
Cohesion: 0.15
Nodes (11): Module, _ensure_compiled(), _FloorplanNet, _gnn_centers(), _load_gnn(), MyOptimizer, Tensor, Mirror of FloorplanNet in iccad2026contest/training_example.py.     Kept here s (+3 more)

### Community 16 - "Portfolio Wrapper"
Cohesion: 0.18
Nodes (13): python_sa_solve(), Pure-Python fallback SA (same as previous v4 approach)., _band_env(), _ensure_compiled(), MyOptimizer, Tensor, M49: per-case env overlay for every profile subprocess (band-gated     REFINE tr, Run one profile; return positions or None. (+5 more)

### Community 17 - "Violation Debug Tools"
Cohesion: 0.12
Nodes (6): Classify boundary violations for given cases: single vs cluster member, and whet, M39 liveness: count pure-movable clusters (no preplaced member, >=2 movable) tha, One-off: classify ALL violating boundary blocks in the PORTFOLIO output (optimiz, Get violation breakdown for high-cost cases., Quick violation breakdown for top cases., For given cases, run all portfolio profiles; print each profile's proxy value vs

### Community 18 - "Baseline Optimizers & Submission"
Cohesion: 0.14
Nodes (9): FloorplanOptimizer, RandomOptimizer, Base class for floorplanning optimizers., Solve the floorplanning problem.                  Args:             block_cou, Simple random placement baseline., Simulated Annealing baseline optimizer., Validate a submission file., SimulatedAnnealingOptimizer (+1 more)

### Community 19 - "Contest Evaluator Harness"
Cohesion: 0.19
Nodes (9): main(), ContestEvaluator, EvaluationResult, Result for a single validation/test case evaluation., Complete evaluation result., Main evaluation engine., Load optimizer from file., Extract baseline metrics from ground truth. (+1 more)

### Community 20 - "Sweep & Oracle Probes"
Cohesion: 0.21
Nodes (8): cost_of(), M37 MIB-aspect ratio sweep (loads dataset ONCE, unlike repeated profile_vs_portf, _parse_output(), keyfun(), OFFLINE ONLY — oracle-perm ceiling probe (never shipped).  Injects an fp_sol-der, run(), Cache (area,hpwl,vrel,true_cost) for every profile×case ONCE (parallel), then sw, run_one()

### Community 21 - "Contest Problem & Constraints"
Cohesion: 0.20
Nodes (12): Problem C Google Drive Submission, get_training/validation_dataloader, iccad2026_evaluate.py, exp(n/12) Exponential Weighting, Hard Constraints, optimizer_template.py (B*-tree SA baseline), Runtime Normalization, Contest Scoring Formula (+4 more)

### Community 22 - "Placer Design Concepts"
Cohesion: 0.29
Nodes (11): Boundary-aspect Dims (2.50/0.40), Cluster Construction, Compaction Area Lever, constructive.cpp (Constructive Placer), Free-aspect Six Sub-axes (M29-M37), Fixed-frame Greedy Packing, HPWL Push Lever (M14-16/24), MIB Shape Unification (apply_safe_mib_dims) (+3 more)

### Community 23 - "RuntimeFactor Lever (RF axis)"
Cohesion: 0.35
Nodes (11): M41 Swap-profile Cut, M42 Big-redundant Profile Cut, M45 Band Tiers + Cores-adaptive, M46 Hot-path Exact Speedup, M47 Wrapper Overhead Fix, M48 Submission Hardening, M49 Measured REFINE Band Cut, optimizer_constructive.py (Portfolio Wrapper) (+3 more)

### Community 24 - "Block Data Model I"
Cohesion: 0.18
Nodes (11): Block, area, boundary, cluster, is_fixed, is_preplaced, mib, th (+3 more)

### Community 25 - "Block Data Model II"
Cohesion: 0.18
Nodes (11): Block, area, boundary, cluster, is_fixed, is_preplaced, mib, th (+3 more)

### Community 26 - "Input Serialization & Probes"
Cohesion: 0.22
Nodes (8): main(), metrics(), Compare Python-prototype compaction vs C++ in-binary compaction on the SAME star, §3 probe: measure how much wall-time PUSH_JUMP / PUSH_SWAP cost on BIG cases, an, Serialize problem data to the text format expected by optimizer_claude.cpp., _serialize_input(), _binary_runs(), True iff _BIN executes a trivial 1-block case end-to-end (M48). Catches     bina

### Community 27 - "M46 Stage Probe"
Cohesion: 0.29
Nodes (10): mode_diff(), mode_timing(), mode_verify(), mode_verify100(), prep(), M46 stage-budget probe (OFFLINE, never shipped).  Where does the wall-setter run, Full-dataset byte-compare over 6 path-representative profiles (base /     aspect, (tag, env) differential variants for one profile dict. (+2 more)

### Community 28 - "Block Data Model III"
Cohesion: 0.18
Nodes (11): Block, area, boundary, cluster, is_fixed, is_preplaced, mib, th (+3 more)

### Community 29 - "Profile Audit (M25)"
Cohesion: 0.24
Nodes (6): cost(), M25 audit: per-profile win tally / leave-one-out / runtime over the live pool., Deployed selector: _RH=1.4 proxy, first-best in pool order (wrapper parity)., run_one(), select(), total()

### Community 30 - "Reconstruction Paradigm & Ceilings"
Cohesion: 0.25
Nodes (9): Per-case Cost / Total Score Formula, Reconstruction Paradigm, Reconstruction RED-confirmed (M40), Teammate 1.0322 Label-leak Oracle, FloorplanNetV2 (Supervised MSE on fp_sol), FloorplanNetV3 (BL Ranking), Ill-posed One-to-Many Problem, Oracle-perm BL Packer Ceiling (+1 more)

### Community 31 - "HPWL Push Debug"
Cohesion: 0.31
Nodes (8): _adj(), _bbox(), main(), push(), Prototype: post-placement HPWL slack push on the PORTFOLIO's output positions., Weighted L1 median (minimiser of sum w*|x-t|) over [(t, w)]., block -> [(neighbor, w)], block -> [(pin, w)], _wmedian()

### Community 32 - "FloorSet & Intel Datasets"
Cohesion: 0.32
Nodes (8): Intel Test Suite (Lite, optimal metrics), Intel Test Suite (Prime, optimal metrics), Intel Test Layout Images (100 validation), FloorSet Dataset, FloorSet arXiv:2405.05480 (Mallappa 2024), FloorSet-Lite, FloorSet-Prime, Intel Test Dataset (200 static circuits)

### Community 34 - "Portfolio Ceiling Probe"
Cohesion: 0.38
Nodes (6): compute_total_score(), Compute exponentially weighted average score.      Total Score = Σ Cost[i] · e, oracle(), proxy_pick(), proxy_score(), Constructive-portfolio ceiling + offline proxy search.  Runs several env-var pro

### Community 35 - "M48 Cold-start Dry-run"
Cohesion: 0.43
Nodes (6): main(), _phase_eval(), _pick_ids(), OFFLINE (never shipped): M48 Beta cold-start dry-run for the submission package., Child-process body: evaluate `opt_path` on `ids`, print PHASECOST lines., _run_phase()

### Community 36 - "Fast Proxy Metrics (M47)"
Cohesion: 0.33
Nodes (6): _hpwl_b2b_fast(), _hpwl_p2b_fast(), _proxy_metrics(), calculate_hpwl_b2b with the tensor rows pre-converted via tolist().     BIT-IDEN, calculate_hpwl_p2b, tolist-converted like _hpwl_b2b_fast (bit-identical)., Baseline-free (area, hpwl, vrel), computed EXACTLY like the harness so the     l

### Community 37 - "Portfolio Proxy Concepts"
Cohesion: 0.40
Nodes (5): %.17g Output Precision Requirement, Baseline-free Portfolio Proxy, Proxy _RH=1.4 Lever, Shapely vrel Proxy Metric, shapely

### Community 38 - "Baseline Generation"
Cohesion: 0.40
Nodes (5): generate_baselines(), main(), print_contest_info(), Generate baseline metrics for all validation cases., Print contest information.

### Community 39 - "Tree Builder Probe"
Cohesion: 0.50
Nodes (4): build_case(), main(), OFFLINE ONLY — M29 Phase B: connectivity-driven builder prototype (never shipped, Greedy connectivity-driven packer with free aspect. Preplaced blocks are     pin

### Community 40 - "Differentiable Training Loss"
Cohesion: 0.50
Nodes (4): FloorplanNet v1 (Unsupervised GCN), compute_training_loss_differentiable, training_example.py, torch (PyTorch)

### Community 42 - "M47 Proxy Equivalence Gate"
Cohesion: 0.83
Nodes (3): _hpwl_b2b_fast(), _hpwl_p2b_fast(), _proxy_metrics_fast()

### Community 43 - "Pool Index Selection"
Cohesion: 0.50
Nodes (4): _effective_cores(), _pool_indices(), Detected parallelism for tier-4 gating. Conservative: logical count     over-est, Kept _PROFILES indices for this case size under the adaptive-pool tiers     (M41

## Knowledge Gaps
- **153 isolated node(s):** `i`, `j`, `w`, `area`, `is_fixed` (+148 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FloorplanOptimizer` connect `Baseline Optimizers & Submission` to `Skyline Packing`, `Solution Evaluation (HPWL/Area)`, `B*-tree SA Template`, `Legacy SA Wrapper (GNN)`, `Portfolio Wrapper`, `Violation Debug Tools`, `Contest Evaluator Harness`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `ContestEvaluator` connect `Contest Evaluator Harness` to `M48 Cold-start Dry-run`, `Baseline Generation`, `Tree Builder Probe`, `Sequence-pair Packer Probe`, `Proxy Analysis & Debug Tools`, `Violation Debug Tools`, `Baseline Optimizers & Submission`, `Input Serialization & Probes`, `HPWL Push Debug`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `BStarTree` connect `B*-tree SA Template` to `Baseline Optimizers & Submission`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `evaluate_solution()` (e.g. with `main()` and `main()`) actually correct?**
  _`evaluate_solution()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Per-case violation breakdown for constructive.exe. Re-runs each case through eva`, `i`, `j` to the rest of the system?**
  _310 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `M46 Probe Placer` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Should `Constructive Placer (C++)` be split into smaller, more focused modules?**
  _Cohesion score 0.060240963855421686 - nodes in this community are weakly interconnected._