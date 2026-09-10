# Alpha submission (archived 2026-09-10)

**This file and `official_result/` are not part of the submission.** They were added at
close-out to record what the six files next to them are and how they scored. The six
files themselves are the Alpha artefact byte-for-byte, copied from
`Downloads/ICCAD2026_FloorSet/alpha_test_submission/`, which was outside the repository
and would have been lost when the machine was cleared.

## What it scored

Official Alpha result: **1.0286, rank 3**. That is `raw 1.4528` (the M10-era placer:
`%.17g` output + compaction) multiplied by a cost-weighted RuntimeFactor of `0.7081`,
i.e. essentially at the `max(0.7, R^0.3)` floor. The Alpha test set turned out to be
bit-identical to the local validation set.

## official_result/

Also outside the repository until close-out, and also the only copy:

* **`cadc1075.xlsx`** (`a32c8fbbe989b86e6f0444af4dec653e`) -- the organisers' own Alpha
  score sheet. One data row, and it is the primary evidence for the number above:

  ```
  Sub ID     Total Score   Feasible   Total Runtime(s)
  cadc1075   1.0286        100        96.4
  ```

* **`cadc1075_results.json`** (`e69f987c2f9728065c5633113248b952`) -- the per-case
  decomposition, from a local run of the official evaluator against this package
  (`submission_name: my_optimizer`, `total_score: 1.4527876342862842`, stamped
  2026-07-14, so a re-run rather than the submission-time run). This is the raw side
  of the same result: `1.4528 x 0.7081 = 1.0286`.

The 96.4 s across 100 cases is worth keeping in view. The RuntimeFactor was already
at its `max(0.7, R^0.3)` floor at Alpha, which is why the entire M41-M50 runtime
programme could later be shown to have been fully cashed in -- and why, much later,
the Beta leaderboard's rank-1 team turned out to be running 169 s.

## Identity

```
6d80cd4d55cce3a6093cdd80d5c18917  constructive.cpp
4c9d477a5b2a87ff7153a5ff6209ee98  floorplan_gnn.pth
c0a889cc9cc9ddba47289068c755fd49  my_optimizer.py
9f35dfcb018a8fd481126091582441d4  optimizer_claude.cpp
468a0afbb1995e3d6b59cd0b01bc7180  optimizer_claude.py
b70b0d8110080a995fbd9e3e6715dea5  requirements.txt
```

`alpha_test_submission/** -text` in `.gitattributes` keeps these byte-exact across
checkouts, for the same reason `build_submission/**` has that rule: core.autocrlf is
true in this repo and a CRLF rewrite would change every md5 while looking correct.

## How it differs from the Final packages

* **Entry point is `my_optimizer.py`**, not `op_wrapper.py`. The organisers renamed the
  required entry between rounds; the Final packages all use `op_wrapper.py`.
* **`floorplan_gnn.pth` is dead weight.** It is a leftover from the GNN line that the
  ML ledger later closed. Shipping an unused binary became an explicit DQ risk in the
  Final rules ("no unused large binaries"), which is why no later package carries it.
* **`requirements.txt` lists matplotlib / tqdm / requests and no scipy.** The Final
  list is `torch / numpy / shapely / scipy`, because the LP arrived at L114 and the
  organisers' Beta report named scipy directly.
* **No `bin/constructive_linux`.** Alpha compiled `constructive.cpp` on the grader;
  the bundled-ELF-first layout only arrived at M67-C.

`optimizer_claude.py` / `.cpp` are the older SA line, kept in the package as the
fallback path of that era.
