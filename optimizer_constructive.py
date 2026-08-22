#!/usr/bin/env python3
"""
Constructive-placer PORTFOLIO wrapper (M51 state, 2026-07-10).

Drives constructive.exe (C++ constraint-aware constructive floorplanner:
fixed-outline greedy packing + compaction/refine/HPWL-push post-passes). Runs
41 deterministic profiles in parallel and selects the best with a
BASELINE-FREE proxy of the contest cost:

    cost  = (1 + 0.5*(agap + hgap)) * exp(2*vrel) * max(0.7, R^0.3)
    proxy = (area/Â + _RH*hpwl/hmin) * exp(2*vrel)   (Â = 1.035*ΣblockArea,
             hmin = min hpwl over profiles, _RH = 1.4)

vrel is computed shapely-exact like the harness (_proxy_metrics); constructive
is deterministic — no SA timing noise — so the proxy has matched the oracle
ceiling since M13 (re-confirmed M31/M43: zero score leaked to selection).

Official local eval = 1.3265 (M51) — an RF=1.0 FICTION: the local harness
forces the RuntimeFactor term to 1.0 while the real score multiplies each case
by max(0.7,(t/median)^0.3). The default-ON adaptive tiers exploit that hidden
axis: M41/M42/M45 pool cuts in _pool_indices() plus the M49/M50 REFINE band
truncation in _band_env(). ICCAD_ADAPTIVE_POOL=0 restores the full 41-profile
quality-best pool (1.3248, full REFINE); ICCAD_L1_POOL=1 (offline only, pair
with ICCAD_ADAPTIVE_POOL=0) extends it to ~84 profiles for the M53 L1 quality
anchor (1.3176) — never the submission shape. M48 hardening: compile chain +
1-block binary smoke, and solve() degrades exception -> python SA -> trivial
hard-feasible row layout.

ICCAD_CONSTRUCTIVE_SINGLE=1 runs only the FIRST profile (free_aspect — the
empty base profile was M33-pruned). ICCAD_CONSTRUCTIVE_BIN overrides the
binary path. Per-knob details: CLAUDE.md.
"""
import concurrent.futures
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import FloorplanOptimizer
try:
    from optimizer_claude import _serialize_input, _parse_output, python_sa_solve
except ImportError:
    pass  # merged single-file form (M67-B): these are defined at the file tail

try:
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _unary_union
    _SHAPELY = True
except Exception:
    _SHAPELY = False

_BIN = Path(os.environ.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive.exe")))

# 41 active profiles (M51 state; #40 fa22_fc_pin_tight_wire is the newest).
# Every add was validated by profile_vs_portfolio.py / portfolio_ceiling.py
# (>0.05% oracle-min bar); the per-milestone rationale lives in the dated
# comment blocks below. Adding profiles is downside-protected: the proxy picks
# per-case, so a never-best profile costs only runtime (and, since M41,
# RuntimeFactor — check per-profile cpu before adding heavy ones).
#
# M25 audit (profile_audit.py, 2026-06-12, on the M24 jump binary): 18 profiles
# with 0 proxy wins AND leave-one-out dTotal == 0 over all 100 cases are pruned
# below (tagged [M25-pruned]; kept as comments — re-add for hidden-test diversity
# if ever desired). Pruned-pool selection is bit-identical (1.3862). Note the
# per-case wall is dominated by the OS16 profiles' own runtime (max term), so
# pruning cheap knob profiles trims total CPU/contention, not the wall model.
_PROFILES: List[Dict[str, str]] = [
    # [M33-pruned: 0 wins, LOO 0; cluster/free stacks dominate the empty base] {},  # base
    {"ICCAD_FREE_ASPECT": "1"},                                               # free_aspect (M29: per-block interior-single aspect search; wins n=118/97/82 large cases)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_MULT": "2.0"},                     # free_aspect_wire
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # free_gm_wt_wire (M29: +0.159% over 1.3814)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # free_tight_wire (M29: +0.148% over 1.3814)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # free_pin_wt_wire (M30: +0.233% oracle-min; wins high-weight 98 n=119 / 95 / 89; pin-seeded BFS order + free interior aspect, wall-safe ~8s max)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # free_gm_tight_wire (M30: +0.134% oracle-min; case 95 n=116 1.2534->1.2329 + 65/40; GM median-seed + tight frame + free, wall-safe)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # free_pin_tight_wire (M30 r2: +0.405% incr over the 40-prof pool; case 95 n=116 1.2400->1.1967 (0.248%) + 73 n=94 + 87 n=108; PIN-seeded BFS order + tight frame + free aspect, single-pass wall-safe)
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # free_gm_pin_wt_wire (M30 r2: +0.284% incr; case 98 n=119 1.3704->1.3323 (0.280%); GM median-seed + PIN order + free, complements free_pin_tight (95 vs 98))
    # [M30-pruned: 0 wins, LOO 0] {"ICCAD_WIRE_MULT": "2.0"},                  # wire_hi
    # [M30r2-pruned: 0 wins, LOO 0] {"ICCAD_ANCHOR_W": "0.04"},               # anc_lo
    # [M30r2-pruned: 0 wins, LOO 0] {"ICCAD_WIRE_MULT": "0.5", "ICCAD_ANCHOR_W": "0.20"},  # area_lean
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286"},                   # aspect_hi
    # [M34-pruned: 0 wins, LOO 0; per-member FREE_CLUSTER wide ratios reach its hard cases] {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},  # aspect_xhi
    # M32 (2026-06-19): DECOUPLED boundary aspect — high LR with TB left at the
    # DEFAULT 0.40. The coupled aspect_* profiles above always pair a high LR with a
    # low TB, which over-squishes the TOP/BOTTOM blocks and misses cases whose TB
    # blocks want to stay near 0.40. profile_vs_portfolio oracle-min vs the 38-prof
    # pool: LR=4.5 alone +0.186% (hard case 85 n=106 1.6091->1.5364 + 71/43); LR=3.0
    # owns case 71 (->1.2760); TB=0.8 +0.029% (case 49 ->1.6144); TB=0.667 (67/52).
    # NOTE: per-block FREE_BOUNDARY aspect SEARCH was DEAD (0.000%) — the greedy local
    # area term avoids the wide LR shape that edge-capacity needs, so the win requires
    # a UNIFORM (profile-level) aspect, not a per-block one. Cheap (no free/OS search)
    # -> wall-safe; downside-protected by the proxy.
    {"ICCAD_LR_ASPECT": "4.5"},                                               # lr45 (M32: hard case 85 + 71/43)
    # [M34-pruned: 0 wins, LOO 0; case 71 now won by fc_pin_tight ->1.2508] {"ICCAD_LR_ASPECT": "3.0"},  # lr30 (M32: case 71 ->1.2760)
    {"ICCAD_TB_ASPECT": "0.8"},                                               # tb08 (M32: case 49)
    {"ICCAD_TB_ASPECT": "0.667"},                                             # tb0667 (M32: case 67/52)
    # M33 (2026-06-19): cluster-member UNIFORM aspect — reshape pure-movable INTERIOR
    # cluster members to a fixed w/h before make_group_item packs the compound, attacking
    # cases whose cluster-internal packing wants non-square members. profile_vs_portfolio
    # oracle-min vs the 39-prof pool: ca2.0 +0.139% (case 85 n=106 1.5364->1.5096 + 59/64),
    # ca0.6 +0.094% (case 96 n=117 1.3112->1.3005), ca1.25 +0.063% (case 82 n=103 ->1.4197).
    # Complementary on distinct high-weight cases (85/96/82). Cheap (no search) -> wall-safe.
    {"ICCAD_CLUSTER_ASPECT": "2.0"},                                          # ca_wide (M33: case 85 + 59/64/89)
    {"ICCAD_CLUSTER_ASPECT": "0.6"},                                          # ca_tall (M33: case 96 n=117)
    {"ICCAD_CLUSTER_ASPECT": "1.25"},                                         # ca_125 (M33: case 82 n=103)
    # M33 r2: stack cluster-aspect with the free_pin order family — cluster reshape +
    # interior-single free aspect + PIN-seeded BFS order + wire compounds like M30's
    # free_pin (+0.327% incr over the 42-prof pool). FREE_ASPECT adds per-block search
    # (~n) but stays under the OS16 wall -> verify est_wall via profile_audit.
    {"ICCAD_CLUSTER_ASPECT": "2.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # ca_free_pin_wt_wire (M33 r2: +0.327%)
    {"ICCAD_CLUSTER_ASPECT": "0.6", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # ca_tall_free_pin_wt_wire (M33 r3: +0.226% incr over the wide stack; tall members complement wide on distinct cases)
    # M33 r4: tight-frame + GM variants of the cluster stacks (mirrors M30's free_pin_tight
    # being the dominant single + free_gm_pin the complement). Tight frame = fewer/tighter
    # scales -> not slower (~9s max, wall-safe). wide-tight is the biggest single (+0.582%).
    {"ICCAD_CLUSTER_ASPECT": "2.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # ca_free_pin_tight_wire (M33 r4: +0.582% incr; wide members + tight frame)
    {"ICCAD_CLUSTER_ASPECT": "0.6", "ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # ca_tall_free_gm_pin_wt_wire (M33 r4: +0.157% incr; tall members + GM seed, complements wide-tight)
    # M33 r5: WIDER cluster ratios in the tight-stack context are far stronger than the
    # uniform-sweep 2.0 cap suggested (stacking with tight+free+pin AMPLIFIES the aspect
    # effect). ca3.0 is a sharp resonant peak (+1.105% incr; 3.5/4.0/5.0 only 0.3-0.4%):
    # concentrated on the highest-weight large cases — 89 n=110 1.7292->1.6142, 98 n=119,
    # 90 n=111, 87 n=108. ca0.4 extreme-tall (+0.253%) complements on distinct cases.
    {"ICCAD_CLUSTER_ASPECT": "3.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # ca3_free_pin_tight_wire (M33 r5: +1.105% incr; wide 3:1 members crack case 89/98/90/87)
    {"ICCAD_CLUSTER_ASPECT": "0.4", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # ca_xtall_free_pin_tight_wire (M33 r5: +0.253% incr; extreme-tall members)
    # M33 r6: GM-seed variant of the dominant ca3.0 stack — GM median-seed hits different
    # cases than PIN order (like M30's free_gm_pin vs free_pin), +0.173% incr. Saturation:
    # adjacent ratios (2.5/3.25/0.33) were <0.12% and overlap ca3.0/ca0.4 -> stopped here.
    {"ICCAD_CLUSTER_ASPECT": "3.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # ca3_free_gm_pin_tight_wire (M33 r6: +0.173% incr; GM complements PIN on ca3.0)
    # M34 (2026-06-20): PER-MEMBER cluster free-aspect (ICCAD_FREE_CLUSTER) — search EACH
    # pure-movable interior cluster member's aspect independently in make_group_item,
    # arbitrated by the cluster layout-key (fragments,boundary_bad,area,aspect), instead of
    # M33's single UNIFORM ratio per profile. The search is build-time (once per solve, not
    # per-frame like FREE_ASPECT) -> widening the ratio set is wall-free, so it runs WIDE
    # (0.333..4.0) and lets different members of one cluster go tall AND wide in a single
    # profile. Per-member BEATS uniform on the hard high-weight cases: profile_vs_portfolio
    # oracle-min vs the 39-prof M33 pool — PIN variant +0.466% (case 82 n=103 1.4197->1.3633,
    # 89 n=110 1.5954->1.5640 [long-standing worst], 88/79/66), GM-seed variant +0.414%
    # (case 80 n=101 1.4895->1.4024, 83/92/70 — complements PIN like M30's free_gm_pin).
    # Top ratio 4.0 is the resonance (5.0 regressed +0.466->+0.367%). One per-member profile
    # subsumes M33's separate ca3.0/ca0.4 uniform tall/wide profiles (search picks per member).
    {"ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_pin_tight (M34: +0.466%; per-member aspect, cracks 89/88/82/79/66)
    {"ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_gm_pin_tight (M34: +0.414%; GM-seed complement, owns 80/83/92/70)
    # M35 (2026-06-21): PER-MEMBER free-aspect for ANCHORED-cluster movable members
    # (ICCAD_FREE_ANCHORED) — the LAST untried free-aspect residual. M34's FREE_CLUSTER only
    # reshapes PURE-movable clusters; MIXED (preplaced+movable) clusters attach their movable
    # members to the preplaced walls in pack_in_frame's first-pass, where they had kept a fixed
    # square shape. FREE_ANCHORED searches each such member's aspect over FREE_RATIOS jointly
    # with the wall-attach position. PREDICTED DEAD (same M32 FREE_BOUNDARY failure mode:
    # arbitrated by the packing greedy score, no cluster layout-key for wall-attached members) —
    # but the anchored score's extra bp (boundary-penalty) + keep-connected terms flipped it:
    # stacked on fc_pin_tight it is +0.111% oracle-min (7 wins incl. the long-standing WORST case
    # 89 1.5640->1.5538, 65 1.7325->1.6904, 52 1.4071->1.3613, 78). Standalone 0.000%
    # (amplification-only, like M33/M34's cluster sweeps). GM-seed variant was REDUNDANT (wins
    # {89,65,52,69} ⊂ PIN, <= PIN on shared) -> add PIN only. Per-frame search (like FREE_ASPECT)
    # but anchored members are few and concentrated -> stays under the OS16 wall (verify est_wall).
    {"ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_anchored_pin_tight (M35: +0.111% oracle-min; anchored-member aspect search, cracks 89/65/52)
    # M36 (2026-06-21): anchored free-aspect RESIDUAL — two wall-safe wins from the M35
    # anchored axis that M35 left untried. (1) WIDE ratios: FREE_ANCHORED reused FREE_RATIOS
    # (max 2.0); widening to the M34 cluster set (0.333..4.0, resonance 4.0) via the new
    # ICCAD_FREE_ANCHORED_RATIOS knob is WALL-FREE here (anchored members are few -> a 1.6x
    # search multiplier on a small base; n=120 4.3s, unchanged vs M35's 5-ratio). (2) UNGATE
    # boundary: ICCAD_FREE_ANCHORED_BND=1 also lets BOUNDARY anchored members search aspect —
    # PREDICTED DEAD by the M32 per-block FREE_BOUNDARY analogy, but the anchored score's bp +
    # keep-connected terms FLIPPED it AGAIN (cf. M35). The two are COMPLEMENTARY: ungated owns
    # hard case 97 n=118 1.2279->1.1988 (+75/78/65/53); gated owns case 88 n=109 1.4115->1.3852
    # (+70) which UNGATED LOSES. profile_vs_portfolio vs the 38-prof pool: ungated +0.244%,
    # gated +0.104%. DEAD ENDS (not added): OS16 x free-aspect (1c-OS) was +0.630% oracle-min
    # but 48s on n=120 (4x OS16's 12s) — the gain IS the per-frame free search INSIDE the OS
    # swap loop (OS16+PIN+tight WITHOUT free = +0.001%, the cheap part already covered) -> gain
    # structurally coupled to 4x runtime, joins the OS24/OS32/om16 shelf. GM variants: gated+GM
    # +0.008% (dead), ungated+GM +0.067% (only disjoint win case 64 < bar) -> PIN only (cf. M35).
    {"ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_BND": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_anchored_bnd_pin_tight (M36 lead: +0.244%; ungated boundary + wide anchored ratios, owns hard case 97 ->1.1988)
    # M37 (2026-06-22): the LAST untried aspect sub-axis — MIB member shape. apply_safe_mib_dims
    # unifies no-master MIB groups to a sqrt(avg) SQUARE; ICCAD_MIB_ASPECT=r reshapes that shared
    # square to a shared rectangle of the SAME area (MIB violation stays 0, all-interior gated).
    # 14/100 cases have a reshapeable group. PREDICTED WEAK (uniform, tiny groups) but — M35/M36
    # lesson a THIRD time — it FLIPPED when stacked on the M36 anchored recipe at WIDE ratio 5.0:
    # +0.110% oracle-min, cracking the long-standing WORST case 89 n=110 1.5538->1.5232 (the wide
    # MIB block nestles into 89's preplaced-boundary geometry, same family as M34's wide cluster
    # members) + case 61 1.3134->1.3030. Resonance at 5.0 (4.0=+0.098%, 6.0=+0.074% regress);
    # pin_tight / gm_pin stacks DEAD (89 is won only via the anchored knobs -> MIB compounds there);
    # tall side (0.25 -> case 79 +0.027%) below the 0.05% bar -> not added (build-time reshape, wall-free).
    {"ICCAD_MIB_ASPECT": "5.0", "ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_BND": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_anchored_mib_pin_tight (M37: +0.110% oracle-min; wide MIB reshape on the anchored recipe cracks worst case 89 ->1.5232 + 61)
    {"ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fc_anchored_wide_pin_tight (M36 complement: +0.104%; gated wide anchored ratios, owns case 88 ->1.3852 + 70 which ungated loses)
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_WIRE_MULT": "2.0"},  # asp_wire
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},                   # aspect_v7
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},                 # aspect_v10  [M25-pruned]
    # [M34-pruned: 0 wins, LOO 0] {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # asp7_wire
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},   # asp5_anclo
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33"},                           # frame_tall  [M25-pruned] (its combos below carry the axis)
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20"},  # frame_tight
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_SCALES": "1.04,1.07,1.10,1.13,1.16,1.35,1.65,2.10"},  # frame_fine
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10", "ICCAD_WIRE_MULT": "2.0"},  # asp10_wire  [M25-pruned]
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},  # tall_asp5
    # {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_WIRE_MULT": "2.0"},   # asp5_wire  [M25-pruned]
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_WIRE_MULT": "2.0"},  # tall_wire
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04"},  # asp7_anclo  [M25-pruned]
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_ANCHOR_W": "0.04"},  # asp_anclo
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},  # tall_asp7  [M25-pruned]
    # {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},   # tight_asp5  [M25-pruned]
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "3.0"},  # asp7_wirex3
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},  # tall_asp10  [M25-pruned]
    # [M30-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_ANCHOR_W": "0.04"},  # tall_anclo
    # [M34-pruned: 0 wins, LOO 0] {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # asp5_all
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10", "ICCAD_WIRE_MULT": "3.0"},  # asp10_wirex3  [M25-pruned]
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # asp7_all  [M25-pruned]
    # [M32-pruned: 0 wins, LOO 0; subsumed by decoupled lr45/lr30] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp5_wire
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp7_wire  [M25-pruned]
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},   # tall_asp5_anc  [M25-pruned]
    {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04"},  # tall_asp7_anc
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "3.0"},  # asp7_all_x3  [M25-pruned]
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20",  "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "3.0"},  # asp5_all_x3
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp7_all  [M25-pruned]
    # [M36-pruned: 0 wins, LOO 0; absorbed once the M36 anchored profiles took its cases] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20",  "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp5_all
    # {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # tight_asp7_wire  [M25-pruned]
    # [M30r2-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # tight_wire
    # M13: narrower-than-frame_tall outlines (aspect 0.55-0.28) attack the systematic
    # horizontal dead space (dbg_area: w/wb~1.3-1.5, h/hb~1.0 -> we pack too wide).
    # Wins the highest-weight cases 98 (n=119) and 87 that frame_tall (0.67-0.33)
    # didn't reach. Downside-protected by the proxy.
    # [M33-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.55,0.45,0.35,0.28"},  # narrow
    # [M30-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.55,0.45,0.35,0.28", "ICCAD_WIRE_MULT": "2.0", "ICCAD_ANCHOR_W": "0.04"}, # narrow_wire_anc
    # {"ICCAD_FRAME_ASPECTS": "0.55,0.45,0.35,0.28", "ICCAD_WIRE_MULT": "2.0"},                         # narrow_wire  [M25-pruned]
    # {"ICCAD_FRAME_ASPECTS": "0.55,0.45,0.35,0.28", "ICCAD_ANCHOR_W": "0.04"},                         # narrow_anc  [M25-pruned]
    # M17: WIRE_TIEBREAK pack-order axis — boundary priority (bscore) intact, but
    # inside each bscore class the most-connected items are placed first, so the
    # greedy wire term sees its heavy neighbours early. Plain WIRE_ORDER (wire as
    # the FIRST key) failed (vBd 390); the tie-break variant keeps vBd. Offline scan
    # (profile_vs_portfolio): wtb_tall_wire wins case 79 (densest uniform graph,
    # 1.706->1.597, the worst single-case hgap) + 89/53/16; wtb_wire wins the
    # highest-weight case 98 (1.4502->1.4413) + 63. Other combos (narrow/LR5/W3)
    # were dominated by these two and are not worth the runtime.
    # [M30r2-pruned: 0 wins, LOO 0] {"ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},             # wtb_wire
    # [M32-pruned: 0 wins, LOO 0; subsumed by decoupled lr45/lr30] {"ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33"}, # wtb_tall_wire
    # M18: WIRE_BFS pack-order axis — bscore classes intact, but inside each class
    # items are emitted greedily by largest edge weight into the already-ordered
    # set + preplaced blocks (BFS over the connectivity graph), so the greedy wire
    # term sees the placed side of an item's heavy edges. Layers over WT (tie order)
    # and frame shapes. Offline scan: bfs_wt_wire +0.316% oracle-min (wins the
    # highest-weight case 98 1.4413->1.4221 plus 95/91/50); bfs_tall_wire +0.251%
    # with ZERO overlap (79 1.597->1.525, 86/74/97/89). wtb_tall_anc adds the
    # otherwise-uncovered case 87 (+0.071%). Plain BFS/BFS+narrow were dominated.
    # [M33-pruned: 0 wins, LOO 0; subsumed by the cluster+free+pin stacks] {"ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # bfs_wt_wire
    # [M32-pruned: 0 wins, LOO 0; subsumed by decoupled lr45/lr30] {"ICCAD_WIRE_BFS": "1", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_WIRE_MULT": "2.0"},      # bfs_tall_wire
    # {"ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_ANCHOR_W": "0.04"}, # wtb_tall_anc  [M25-pruned] (case 87 superseded since M18)
    # M19: BFS_PIN seeds the BFS attachment with p2b pin weights too (pins are
    # fixed anchors exactly like preplaced blocks). bfs_pin_wt_wire +0.269%
    # oracle-min: re-breaks case 95 (1.2995->1.2767) and takes 94 (1.3656->1.3411)
    # + 64. bfs_tight_wire +0.061%: case 91 (1.3848->1.3712, untouched by PIN)
    # + small-n cases. BFS+anc/PIN+W2/PIN+tall were dominated by these two.
    # [M33-pruned: 0 wins, LOO 0; subsumed by the cluster+free+pin stacks] {"ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # bfs_pin_wt_wire
    {"ICCAD_WIRE_BFS": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},       # bfs_tight_wire
    # M20: ORDER_SWAP=K — before refinement, greedy pair-swap hill-climb on the
    # pack order over the top-K total_wire items (pack-once comparisons, strict
    # layout_score improvement only). A jump move greedy ordering + force-directed
    # refinement can't make. On the strongest ordering it re-breaks case 94
    # (1.3411->1.3128) and 98 (1.4221->1.4118) + 79/50/28/9 (+0.252% oracle-min).
    # Plain OS8 (+0.030%) and OS8+tall (+0.025%) won only low-weight cases — the
    # swap needs the PIN+WT order as its starting point to matter at high weight.
    {"ICCAD_ORDER_SWAP": "8", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # os_pin_wt_wire
    # M21: ORDER_SWAP combo sweep. A larger swap pool (K=16, 120 pairs) on the
    # strongest ordering is the biggest single profile since M18 (+0.460%
    # oracle-min): re-breaks the highest-weight case 98 (1.4118->1.3841) and 79
    # (1.5135->1.4219) + 82/89/91. OS8 on BFS+WT *without* PIN (+0.262%) wins
    # 86/95/97; OS8 on BFS+tight (+0.221%) is the first profile to dent hard
    # case 85 (1.6606->1.6255) + 42/40. Wins are near-disjoint. OS12 (+0.157%)
    # and OS16+tall (+0.056%) were dominated by these three; OS8+WT+tall +0.016%.
    {"ICCAD_ORDER_SWAP": "16", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # os16_pin_wt_wire
    # [M30-pruned: 0 wins, LOO 0; frees ~2s cpu] {"ICCAD_ORDER_SWAP": "8", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # os_bfs_wt_wire
    {"ICCAD_ORDER_SWAP": "8", "ICCAD_WIRE_BFS": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},       # os_bfs_tight_wire
    # M22: K-axis follow-up. On the strongest (PIN+WT) order K saturates at 16
    # for the high-weight cases (98/79/82 absent from both OS24 and OS32 wins);
    # K=16 transplanted onto the two M21 no-PIN orders lands borderline-but-
    # disjoint wins and is shipped: os16_bfs_wt (+0.056%: 86/82 deepened + 50)
    # and os16_bfs_tight (+0.057%: new case 62 1.6214->1.5248 + 55/51).
    # NOT shipped (runtime risk, not score): OS32 on PIN+WT (+0.062%: case 71
    # 1.3187, deepest 89 1.8093, 57/21/35) and OS24 (+0.041%: the ONLY profile
    # reaching the long-unharvested case 66, 1.4450->1.3989). Both verified live
    # (full 4-profile portfolio = 1.3979, matching the scan exactly) but they
    # cost 58s/32s CPU on n=120 and pushed avg runtime 8.4->21.9s/case — the
    # official RuntimeFactor uses the cross-submission median (teammate
    # portfolio ~11s is our only reference), so a 2.6x runtime jump risks a
    # 20-35% cost penalty for a 0.04% score gain. Re-add if the runtime rule
    # turns out lenient.
    {"ICCAD_ORDER_SWAP": "16", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},                        # os16_bfs_wt_wire
    {"ICCAD_ORDER_SWAP": "16", "ICCAD_WIRE_BFS": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},       # os16_bfs_tight_wire
    # M23: ORDER_MOVE=K — relocation hill-climb on the pack order (pull one top-K
    # wire item out, insert it at another top-K slot; the segment between shifts).
    # A structural jump ORDER_SWAP can't reach (swap keeps everyone else fixed).
    # om8_pin_wt_wire +0.041% oracle-min, all wins realized live: deepest case 89
    # yet (1.8155->1.8061, beats even the os32 stand-by 1.8093) + 57 (1.3689) +
    # 26/33. Cheap (56 extra packs/frame, n=120 ~7s).
    # NOT shipped (runtime, same call as the M22 os24/os32 shelving):
    # om16_bfs_wt_wire (OM16+BFS+WT, no PIN) +0.148% — first kill of case 96
    # (n=117, 1.3336->1.3160), finally harvests case 66 (1.4378->1.3951, deeper
    # than the os24 stand-by) + 91/42/53/17/19 — fully verified live (59-prof =
    # 1.3968, 20/20 predicted wins realized) but costs ~28s cpu on n=120 and
    # pushed avg runtime 8.8->13.5s/case; (13.5/8.8)^0.3 = +13.8% RuntimeFactor
    # exposure for +0.14% score whenever the official median < ~19s. Re-add
    # together with os24/os32 if the runtime rule turns out lenient.
    # M25 UPDATE (2026-06-12, after the M24 jump): om16's edge has mostly been
    # eaten for free — jump already reaches case 96 1.3171 (om16 1.3160) and 66
    # 1.3981 (om16 1.3951). Audit (profile_audit.py): om16 marginal on top of the
    # M24 pool is only +0.065% (1.3862->1.3853) while its own runtime (mean
    # 13.9s, n=120 36s with jump) DOMINATES the per-case wall -> (13.91/7.30)^0.3
    # = +21% RF premium. Pruning cheap knobs cannot fund it (wall is max-term
    # bound). Stand-by demoted: only worth re-adding if the official runtime
    # rule turns out to ignore runtime entirely.
    # Other M23 scan results: K-amplification is order-dependent like OS
    # (K=8->16 pays on BFS+WT 0.026->0.148% but not on PIN+WT 0.041->0.040%);
    # OM12 loses 96/66 entirely (+0.012%, hill-climb path nonlinear in K);
    # OM8+OS16 stacked +0.070% dominated by the two leads; OM8 on BFS+tight
    # 0.000%; CLUSTER_ORD (compound cluster first/last in bscore class) dead on
    # both orderings tried (0.000%).
    {"ICCAD_GUIDE_MED": "1", "ICCAD_ORDER_MOVE": "8", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},   # gm_om8_pin_wt_wire (M26: GUIDE_MED added in-place to M23 om8_pin; case 87->1.3505 + 82, ~free runtime since OM8 already paid)
    # M26: GUIDE_MED candidate injection — add the connectivity-weighted L1-median of
    # an item's placed/guide neighbours as an extra greedy candidate origin (a wire-
    # optimal seed the geometric abutment slots miss; cheap, one extra candidate per
    # item, no extra packs). Shipped two ways:
    #  (1) GUIDE_MED added IN-PLACE to the M23 om8_pin_wt_wire above (gm_om8_pin_wt_wire):
    #      wins the high-weight case 87 deep (1.4106->1.3505, OM8 relocation + median
    #      seed compound) + case 82. Added in-place (not as a 2nd OM8 profile) so the
    #      OM8 runtime is already paid -> ~free; a separate ADD cost +1.3s/case for a
    #      +4.7% RuntimeFactor premium (M22/M23/M25 runtime discipline) and was rejected.
    #  (2) gm_bfs_wt_wire below (no PIN): wins the disjoint case 91 (1.3569->1.3481).
    # Scanned but NOT added (all <0.05% incremental, M25 lean-pool discipline): gm on
    # tight/tall/narrow frames (<=0.015%); a separate ADD of gm_om8 (+1.3s runtime).
    # The other two M26 candidates were DEAD (see CLAUDE.md M26): reframe (re-seed the
    # frame loop from the measured compacted bbox) <=0.035%, redundant with the
    # portfolio's aspect diversity; the oracle-perm ordering probe ceiling was
    # 0.002-0.005% (injecting the PERFECT fp_sol order gains nothing -> the placer, not
    # the pack order, is the bottleneck -> ordering/ML ranking permanently closed).
    # [M30-pruned: 0 wins, LOO 0; superseded by free_gm_wt_wire] {"ICCAD_GUIDE_MED": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # gm_bfs_wt_wire (case 91)
    # M51 (2026-07-10): wide-CLAMPED frame — the default FRAME_ASPECTS caps at 1.8,
    # but any aspect >=2.0 on a preplaced-heavy case clamps (frame_candidates:
    # w/h floors at pre_*+MARGIN / max_i*+MARGIN) to ONE fixed wide outline the
    # default set never generates. On the heaviest case 99 (n=120, 8.0% weight)
    # that clamped frame beats both layout_score and true cost: 1.3334->1.3084
    # (+0.200% weighted, the largest single-case find since M37), bit-identical
    # across aspects 2.0/2.2/2.5 (clamp-invariant) and across REFINE_ITERS 4/12
    # (valid under the shipped M49 big-band overlay). Host = fc_pin_tight (true
    # cost 1.3084 vs 1.3129 on the M37 anchored host); first three tried frames
    # (1.0/0.75/1.35) stay identical to the default order, only the 4th slot
    # changes. Standalone FRAME_ASPECTS sweeps (wide/mid/no-square/mild-tall)
    # were all ~0.000% — the win only exists stacked (M33-M37 amplification
    # pattern). BP_WEIGHT-down (10000/3000/1000/300, standalone + both hard-case
    # stacks) was 0.000% everywhere -> that axis is closed RED.
    {"ICCAD_FRAME_ASPECTS": "1.0,0.75,1.35,2.2", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # fa22_fc_pin_tight_wire (M51: clamped-wide frame cracks case 99 ->1.3084)
]

# M53 L1 (2026-07-12): score-first quality pool — OFF by default; enable with
# ICCAD_L1_POOL=1, meant to pair with ICCAD_ADAPTIVE_POOL=0 (the offline quality
# baseline for the M53 L2/L3 probes; NEVER the submission profile). Gate off ->
# _PROFILES is bit-identical to shipped, so no rf_score_model/m49 re-gate needed.
# Two parts, both measured 2026-07-12 against the 41-prof POOL=0 baseline 1.3248:
#  (1) _L1_EXTRA: the M36 OS16xfree family (shelved purely for RuntimeFactor:
#      ~48s on n=120). Re-validated via profile_vs_portfolio vs the min(K12,K24)
#      oracle baseline 1.3241: os16 on the fc_anchored_bnd recipe = +0.509%
#      oracle-min (18 wins: 80/73/97/82/78/86/75/66...).
#  (2) a REFINE_ITERS=24 duplicate of every base profile: the GLOBAL override
#      regresses (+0.105%; layout_score!=true-cost poison on cases 84/73), but as
#      a portfolio tier the proxy keeps per-case winners -> oracle -0.053%
#      (96/77/72/70/27/64). K=48 adds only -0.010% over K=24 -> dropped;
#      PUSH_PASSES/COMPACT_ITERS=64 are exact no-ops (both loops early-break).
# NOTE: with ICCAD_ADAPTIVE_POOL=1 the swap filter in _pool_indices() drops the
# OS16 extras anyway, but the K24 duplicates would not be dropped — always run
# this gate together with ICCAD_ADAPTIVE_POOL=0 (and ICCAD_PROFILE_TIMEOUT high:
# the ~84-way pool oversubscribes cores, stretching per-profile wall past 120s).
_L1_EXTRA: List[Dict[str, str]] = [
    {"ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_BND": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0", "ICCAD_ORDER_SWAP": "16"},  # os16_fc_anchored_bnd_pin_tight (M36 1c+OS, L1 re-val: +0.509%; wins 80/73/97/82/78/86/75/66)
    {"ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0", "ICCAD_ORDER_SWAP": "16"},  # os16_fc_pin_tight (L1: +0.151% incr over P1; disjoint wins 95/90/79/64/35)
]
if os.environ.get("ICCAD_L1_POOL", "0") == "1":
    _L1_BASE = list(_PROFILES)
    _PROFILES.extend(_L1_EXTRA)
    _PROFILES.extend(dict(p, ICCAD_REFINE_ITERS="24") for p in _L1_BASE)

# M72 (2026-07-30, ported from the teammate's b716753 + our own fix): the SAME six
# cluster-boundary knobs M71 exports GLOBALLY, but as four extra PROFILES instead.
# OFF by default (ICCAD_M55_POOL=1 enables) -> _pool_indices() is bit-identical to
# shipped, so no rf_score_model/m49 re-gate is needed to carry the knob.
#
# WHY a tier and not more global knobs: M71 (EXPOSE+EDGE_PACK on every profile)
# buys -1.5894% but REGRESSES 17/100 cases - in global mode a regressing case has
# nowhere to escape, because every candidate in the pool carries the knobs. The
# teammate measured a per-case 2-way oracle between the knob-off and knob-on
# portfolios at 1.299157, i.e. most of the residual value is in ESCAPING those 17.
# A pool tier gives the proxy both worlds per case.
#   knob-on-always (our shipped M71)          1.305390   56 better / 17 WORSE
#   4-profile tier on a knob-off base (M72)   1.306635   31 better /  0 worse
#   per-case 2-way oracle                     1.299157
# Their four recipes mirror the donor M55 block (teammate_m43 idx 43-46) on this
# engine's host recipe; profile_vs_portfolio oracle-min vs the knob-off 41-pool:
# +1.287% / +1.518% / +1.531% / +1.497% (25-30x the 0.05% bar), and case 89 - the
# highest-cost case in the set - moves 1.5232 -> 1.3707, i.e. NOT the M39
# FREE_CLUSTER_BND pattern where the hard cases never moved.
# Appended at the END on purpose: _BIG_REDUNDANT_IDX / _M45_BAND_DROP /
# _M45_LOWCORE_DROP are index-based frozensets over 0..40, so indices >=41 cannot
# disturb them. NEVER insert or reorder _PROFILES.
# Extended UNCONDITIONALLY (indices stay stable) and gated at CALL time inside
# _pool_indices(): m67_oos_probe.py flips env vars AFTER importing this module, so
# an import-time gate would silently be a no-op and the probe would measure a fake
# "zero difference" (a false negative). Same convention as ICCAD_M67F_RESTORE.
_M55_EXTRA: List[Dict[str, str]] = [
    {"ICCAD_CLUSTER_BND_EXPOSE": "1", "ICCAD_CLUSTER_BND_CORNER": "1", "ICCAD_CLUSTER_BND_PERMUTE": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # m55_cluster_bnd_permute (+1.287%)
    {"ICCAD_CLUSTER_BND_EXPOSE": "1", "ICCAD_CLUSTER_BND_CORNER": "1", "ICCAD_CLUSTER_BND_PERMUTE": "1", "ICCAD_CLUSTER_BND_EDGE_PACK": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # m55_cluster_bnd_edge_pack (+1.518%)
    {"ICCAD_CLUSTER_BND_EXPOSE": "1", "ICCAD_CLUSTER_BND_CORNER": "1", "ICCAD_CLUSTER_BND_PERMUTE": "1", "ICCAD_FRAME_SCALES": "1.00,1.025,1.05,1.10", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # m55_area_tight_cluster_bnd (+1.531%)
    {"ICCAD_HPWL_SAFE_CLUSTER_SLIDE": "1", "ICCAD_CLUSTER_BND_EXPOSE": "1", "ICCAD_CLUSTER_BND_CORNER": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # m55_hpwl_safe_cluster_slide (+1.497%)
]
_M55_BASE_LEN = len(_PROFILES)
_PROFILES.extend(_M55_EXTRA)
_M55_IDX = frozenset(range(_M55_BASE_LEN, len(_PROFILES)))

# M76 (2026-08-01): the teammate's M73 knob-OFF ESCAPE tier (their 7403758),
# ported for measurement under OUR M74 baseline. OFFLINE/EXPERIMENTAL, default OFF.
#
# MECHANISM. Shipped M71 rides a per-profile overlay applied to EVERY index in the
# pool (see _m71_env), so a case the two cluster knobs hurt has nowhere to escape
# to. The tier appends knob-OFF DUPLICATES of chosen hosts and _solve_impl
# deliberately skips the M71 overlay on exactly those indices, so the pool carries
# both variants of a host and the proxy arbitrates per case. It is the MIRROR of
# M72 (which added knob-ON profiles to a knob-off base and measured -1.418% OOS
# for us); this direction is the one that matches what we actually shipped.
#
# WHY ALL 41 HOSTS ARE APPENDED. The active subset is chosen at CALL time from
# ICCAD_M73_SRC, not at import: m67_oos_probe.py's arm mechanism does
# os.environ.update() AFTER importing this module, so an import-time source list
# could not be swept and every arm would silently measure the same thing. Trimming
# _M73_ESCAPE to the final set is a SHIP-time step, not a measurement-time one.
#
# The teammate's set was fitted to M71 with the PRE-M74 adaptive constants, on the
# 7 highest weighted-recoverable of the 17 cases M71 regressed (#22/#23/#25 rescue
# 4/7 each, #2 is the only rescuer of case 94). Under M74 all four survive the
# heavy pool (13 profiles @12c, 35 @48c) but only #2/#22/#25 survive the mid pool
# at <=16 cores, because M74's tier-3 drops #23 there.
_M73_ESCAPE: List[Dict[str, str]] = [dict(p) for p in _PROFILES[:_M55_BASE_LEN]]
_M73_BASE = len(_PROFILES)              # _M73_BASE + h == knob-off twin of host h
_M73_IDX = frozenset(range(_M73_BASE, _M73_BASE + len(_M73_ESCAPE)))
_PROFILES.extend(_M73_ESCAPE)
_M73_SRC: Tuple[int, ...] = (2, 22, 23, 25)     # teammate's measured default


def _m73_active() -> FrozenSet[int]:
    """Escape indices this call should add, resolved from the environment.

    Read at CALL time (never at import) for the reason above. A malformed
    ICCAD_M73_SRC falls back to _M73_SRC rather than raising: this runs inside
    solve() on the grader, where an exception would cost the whole case."""
    if os.environ.get("ICCAD_M73_ESCAPE", "0") != "1":
        return frozenset()
    raw = os.environ.get("ICCAD_M73_SRC", "")
    src: Tuple[int, ...] = _M73_SRC
    if raw:
        try:
            src = tuple(int(x) for x in raw.split(",") if x.strip())
        except ValueError:
            src = _M73_SRC
    return frozenset(_M73_BASE + h for h in src if 0 <= h < _M55_BASE_LEN)


# M80 (2026-08-05): the M79 knob-cloud tier. SHIPPED, cores-gated, default ON.
#
# MECHANISM. These are not hand-tuned recipes and not twins of anything: they are
# points drawn by RANDOM JOINT sampling of the ~15-dim env-knob space the shipped
# profiles live in, then greedily selected as FIXED profiles (no per-case choice)
# against the 100-case portfolio. M30/M31 swept this space one knob at a time,
# outward from a hand-stacked recipe, stopping below 0.05% — and saturated at
# <=0.063% per new profile. Joint sampling reaches combinations coordinate-wise
# greedy never visits: the first pick raises BP_WEIGHT, pushes MIB_ASPECT to the
# tall side AND widens the frame scales, three moves the dead-end ledger had each
# independently judged worthless. Single dead does not imply jointly dead.
#
# WHY CORES-GATED. The wall is max(max_i dt_i, sum_i dt_i / cores). At 48 cores
# 100/100 cases are max-setter bound, so K extra profiles that are each cheaper
# than the incumbent max cost ~nothing; at 12 cores the sum term dominates and
# M79 measured dRF +10.614% at K=8 with 100/100 cases getting a higher wall. So
# this rides exactly the M67-F tier-5 bet (>= 40 DETECTED cores, which is an
# upper bound on effective ones) and fails CLOSED the same way.
#
# Appended past _M55_BASE_LEN like _M55_EXTRA/_M73_ESCAPE, which is what keeps
# every offline cache valid: profile_audit / rf_score_model / m67 / m77 all
# anchor their signatures on _PROFILES[:_M55_BASE_LEN]. Indices >= 41 are also
# invisible to the index-based drop sets. Gated at CALL time, never at import.
# Greedy order over an R=256 cloud (seed 79); K=8 chosen OUT OF SAMPLE, not by
# the in-sample curve. Both 240-case samples put a clean elbow at exactly 8: the
# 8th vector is worth +0.195pp (s1) / +0.249pp (s2) of NET and the 9th is worth
# +0.004 / +0.009pp. K=12 would buy +0.019pp more while making the pool 50%
# bigger, and pool size is the exposure if the grader's EFFECTIVE parallelism is
# below its detected count — the same bet tier-5 already carries, so it is not
# worth doubling for 1% of the gain.
# Machine-readable source of truth: m80_vectors.json (which holds all 12),
# asserted verbatim against this prefix by m80_tier_gate.py V5 — build_cloud()
# is seeded but its output depends on the shipped prefix, so without that file
# "#100" is a moving target.
_M80_EXTRA: List[Dict[str, str]] = [
    {"ICCAD_BP_WEIGHT": "274048", "ICCAD_CLUSTER_ASPECT": "3.39", "ICCAD_FRAME_SCALES": "1.00,1.10,1.25,1.45", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.25,0.4,0.6667,1.0,1.5,2.5,4.0", "ICCAD_LR_ASPECT": "2.044", "ICCAD_MIB_ASPECT": "0.2338", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "3.273", "ICCAD_WIRE_TIEBREAK": "1"},  # cloud #100
    {"ICCAD_BFS_PIN": "1", "ICCAD_BP_WEIGHT": "66359.9", "ICCAD_CLUSTER_ASPECT": "1.262", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_FRAME_SCALES": "1.00,1.10,1.25,1.45", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.25,0.4,0.6667,1.0,1.5,2.5,4.0", "ICCAD_TB_ASPECT": "0.3585", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "0.9473"},  # cloud #182
    {"ICCAD_BFS_PIN": "1", "ICCAD_BP_WEIGHT": "63806", "ICCAD_CLUSTER_ASPECT": "3.04888", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_BND": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_SOFT_ASPECT": "1.32324", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "2.0", "ICCAD_WIRE_TIEBREAK": "1"},  # cloud #133
    {"ICCAD_ANCHOR_W": "0.0329875", "ICCAD_BFS_PIN": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_LR_ASPECT": "4.08896", "ICCAD_MIB_ASPECT": "2.29473", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "2.0", "ICCAD_WIRE_TIEBREAK": "1"},  # cloud #0
    {"ICCAD_BFS_PIN": "1", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ANCHORED_BND": "1", "ICCAD_FREE_ANCHORED_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_MIB_ASPECT": "5.0", "ICCAD_SOFT_ASPECT": "0.742103", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "3.14767", "ICCAD_WIRE_TIEBREAK": "1"},  # cloud #224
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.5,0.6667,1.0,1.5,2.0", "ICCAD_SOFT_ASPECT": "1.08607", "ICCAD_WIRE_MULT": "2.0"},  # cloud #198
    {"ICCAD_ANCHOR_W": "0.1131", "ICCAD_CLUSTER_ASPECT": "0.9663", "ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.25,0.4,0.6667,1.0,1.5,2.5,4.0", "ICCAD_GUIDE_MED": "1", "ICCAD_LR_ASPECT": "2.61", "ICCAD_MIB_ASPECT": "3.108", "ICCAD_TB_ASPECT": "0.9931", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "0.6905"},  # cloud #56
    {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_FREE_ANCHORED": "1", "ICCAD_FREE_ASPECT": "1", "ICCAD_FREE_CLUSTER": "1", "ICCAD_FREE_CLUSTER_RATIOS": "0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0", "ICCAD_LR_ASPECT": "4.50602", "ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_MULT": "2.33755", "ICCAD_WIRE_TIEBREAK": "1"},  # cloud #170
]
_M80_BASE = len(_PROFILES)
_M80_IDX = frozenset(range(_M80_BASE, _M80_BASE + len(_M80_EXTRA)))
_PROFILES.extend(_M80_EXTRA)
# Same value as _M67F_CORES_MIN and the same bet, but a separate constant so the
# two tiers can be retuned independently if the Beta box ever reports a real
# effective parallelism.
_M80_CORES_MIN = 40

# L124 (2026-08-12): MIB shape-bucketing TWINS. Eight shipped profiles duplicated
# with ICCAD_MIB_BUCKET=1, appended so the proxy can arbitrate between the two
# behaviours per case instead of one of them being forced on every case.
#
# WHY A TWIN AND NOT A GLOBAL FLAG. L123 measured the global overlay and it
# FLIPPED SIGN across the two disjoint out-of-sample draws: s1 +0.6486%,
# s2 -0.3730%. But the per-case oracle over {on, off} is positive on BOTH
# (s1 +1.5388%, s2 +0.8590%) with the same 34/80 split each time -- the mechanism
# is real, forcing one setting on every case is what failed. The proxy realises
# 68-88% of that oracle (measured, not assumed: M76/M77 showed it is oracle-
# perfect on heterogeneous candidates, which is why no per-case classifier is
# needed -- that is the thing M56 and M79 killed).
#
# WHY THESE EIGHT. Every profile's ON twin was scored on how often it would win
# if all twins existed; these are the top eight by combined s1+s2 tally, and all
# eight have support in BOTH samples. Selecting the set on one sample and scoring
# it on the other transfers at 80-83% (s1-set on s2 +0.4712%, s2-set on s1
# +1.0943%), so this is not an in-sample artefact. Six of the eight are M80 tier
# profiles: the aggressive knob combinations benefit most from bucketing.
#
# WHY APPEND AND NOT REPLACE. Appending is interference-free (replacing would
# also remove the source profile's OFF output, which wins on ~42% of heavy
# cases). 43 -> 51 stays well inside M67-E's free-restore budget, which puts the
# max-bound -> sum-bound crossover at 75-80 profiles, so ΔRF is ~0 at 48 cores.
# Cores-gated for the same reason M80 is: below the gate the pool is sum-bound
# and eight more profiles would cost wall for a benefit that is held-out only
# (in-set is a provable no-op -- all 100 in-set MIB groups already unify, which
# is why in-set MIB has always read 0).
_M124_SRC = (16, 31, 87, 88, 89, 90, 92, 93)
_M124_EXTRA = [dict(_PROFILES[i], ICCAD_MIB_BUCKET="1") for i in _M124_SRC]
_M124_BASE = len(_PROFILES)
_M124_IDX = frozenset(range(_M124_BASE, _M124_BASE + len(_M124_EXTRA)))
_PROFILES.extend(_M124_EXTRA)
_M124_CORES_MIN = 40


# L137 (2026-08-17): the GORDIAN hint as a POOL TIER, not a global overlay.
#
# WHY A TIER. Applied globally (ICCAD_HINT_MODE=1 on every profile) the hint is
# quality-positive -- in-set 48c 1.2284738 -> 1.2279371 (+0.0437%), OOS s1 240
# cases 1.563347 -> 1.561957 (+0.0889%) with hpwl_gap 0.3135 -> 0.3116 and
# area_gap 0.2569 -> 0.2529, both moving the way the mechanism predicts -- but it
# also moves the 48c wall +1.76%, and that cost is NOT spread: case 90 alone is
# 190% of the weighted delta and case 91 another 73% (over 100% because the rest
# get FASTER). Timed on one profile, case 90 is 0.213 -> 0.380s while case 92,
# the biggest quality gain, is 0.674 -> 0.674s unchanged.
#
# A tier removes exactly that cost. At >=40 cores the wall is the max-setter
# (M67-E, 100/100), so a 0.38s hinted profile against case 90's 2.93s wall costs
# nothing on the max term, and the existing profiles keep their exact runtime AND
# their exact output. M76/M77 measured the proxy as oracle-perfect on
# heterogeneous candidates, so an added candidate cannot lose quality -- it can
# only fail to be selected.
#
# Same discipline as M72/M76/M80/M124: appended UNCONDITIONALLY so indices stay
# stable (_BIG_REDUNDANT_IDX / _M45_BAND_DROP are index-based frozensets over
# 0..40), gated at CALL time so a probe that sets the env after import actually
# flips it, and NEVER inserted or reordered.
#
# Sources are four diverse always-in-pool base recipes (<_M55_BASE_LEN so they
# are never dropped by the M55 gate): free_aspect, free_gm_wt_wire,
# free_pin_wt_wire, free_gm_tight_wire -- spanning GUIDE_MED / BFS_PIN / tight
# frames, since the hint changes the anchor term and that interacts with the
# wire and frame knobs.
_L137_SRC = (0, 2, 4, 5)
_L137_EXTRA = [dict(_PROFILES[i], ICCAD_HINT_MODE="1") for i in _L137_SRC]
_L137_BASE = len(_PROFILES)
_L137_IDX = frozenset(range(_L137_BASE, _L137_BASE + len(_L137_EXTRA)))
_PROFILES.extend(_L137_EXTRA)
_L137_CORES_MIN = 40


def _l137_active(block_count: int) -> FrozenSet[int]:
    """Tier indices this call should add. CALL time, never import time.

    Uses _effective_cores_hi() (unknown -> 0) for the same reason M80/M124 do:
    this tier fires at HIGH core counts, so the 9999 sentinel that keeps tier-4
    safe would switch it ON wherever detection fails -- and below the gate the
    pool is sum-bound, where four more profiles DO cost wall.

    Default OFF while the net is unresolved: the quality is measured and the
    runtime is not yet, so this must not ship by accident."""
    if os.environ.get("ICCAD_HINT_POOL", "0") != "1":
        return frozenset()
    if _effective_cores_hi() < _L137_CORES_MIN:
        return frozenset()
    return _L137_IDX


def _m124_active(block_count: int) -> FrozenSet[int]:
    """Tier indices this call should add. Read at CALL time, never at import, so
    a probe that sets the env after importing this module actually flips it.

    Uses _effective_cores_hi() (unknown -> 0) for the same reason M80 and tier-5
    do: this tier fires at HIGH core counts, so the 9999 sentinel that keeps
    tier-4 safe would turn this one ON wherever detection fails.

    ICCAD_M124_TWIN=0 is the kill switch, and it -- not a comparison against
    HEAD -- is the permanent invariant once the tier ships on by default."""
    if os.environ.get("ICCAD_M124_TWIN", "1") == "0":
        return frozenset()
    if _effective_cores_hi() < _M124_CORES_MIN:
        return frozenset()
    return _M124_IDX


def _m80_active(block_count: int) -> FrozenSet[int]:
    """Tier indices this call should add. Read at CALL time (never at import) so
    m67_oos_probe.py's arms, which os.environ.update() AFTER importing this
    module, actually flip it instead of silently measuring a no-op.

    Uses _effective_cores_hi() (unknown -> 0), NOT _effective_cores() (unknown ->
    9999): this tier fires at HIGH core counts, so the 9999 sentinel that keeps
    tier-3/tier-4 safely off would turn this one ON wherever detection fails.
    Both directions must fail to the shipped pool. Malformed ICCAD_M80_MIN_N
    falls back to 0 rather than raising - this runs inside solve() on the grader,
    where an exception costs the whole case.

    SHIPPED ON (kill-switch semantics, like ICCAD_M67F_TIER5): measured
    NET +1.786% / +1.909% on the two disjoint 240-case out-of-sample draws at the
    grader's 48-core pool shape, against a 0.30% bar. ICCAD_M80_TIER=0 disables."""
    if os.environ.get("ICCAD_M80_TIER", "1") == "0":
        return frozenset()
    if _effective_cores_hi() < _M80_CORES_MIN:
        return frozenset()
    try:
        min_n = int(os.environ.get("ICCAD_M80_MIN_N", "0") or 0)
    except ValueError:
        min_n = 0
    return frozenset() if block_count <= min_n else _M80_IDX


# M42 (2026-06-26): 2nd-order RuntimeFactor lever — indices into _PROFILES of the
# BUILD-time profiles that are NEVER the selected winner on any case with n>100
# (per-big-n redundancy; the M41 swap argument applied to the build profiles).
# After M41 drops the 6 swap profiles, the big-case (n>=110, 60% weight) wall is
# sum/cores-bound by the ~20 expensive (~8s) FREE/CA/FC profiles; these 22
# win NO n>100 case, so dropping them there is wall-only (every capped case keeps
# Qcap/Qbase=1.0000 while the heavy wall roughly halves, and all 20 n>100 cases
# stay median-INDEPENDENT WINs). The kept 13 build profiles are the
# cluster/anchored/MIB stacks that structurally dominate big cases + a few
# aspect/free winners. REGENERATE after any _PROFILES edit OR any constructive.exe
# rebuild: rf_score_model.py's M42 section prints the recommended set.
#
# M74 (2026-07-30) REGENERATED under the shipping configuration. The previous set
# came from a 2026-07-10 audit_cache.pkl built on the PRE-M71 binary at REFINE=12;
# M71 moved every position, so that redundancy tally was stale. Still 22 indices
# and still all-win at T=100, but the MEMBERSHIP moved a lot ({1,3,6,18,20} out,
# {8,12,13,21,40} in). Model: local RF=1.0 1.2935, bit-identical to the M41
# post-swap baseline; projected real gain +2.88% @M=11.
_BIG_REDUNDANT_IDX = frozenset({0, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                21, 24, 28, 29, 30, 31, 32, 33, 40})

# M45 (2026-07-02): per-band pool tiers — the M42 redundancy generalized from
# CUMULATIVE ("wins no n>T case") to BAND-scoped ("wins no lo<n<=hi case"), which
# frees profiles that are big-case winners but mid-band dead weight. Sets are
# derived by rf_score_model.py's M45 section under a STRICT selection-preserving
# gate: every band case keeps an EQUAL-cost selection (rel 1e-9) after the drop,
# so quality is validation-identical and the wall-only cut weakly wins for EVERY
# cross-submission median and EVERY core count (no RF-floor caveat). Band edges
# never straddle n=100 (pool composition changes there). REGENERATE both this
# and _BIG_REDUNDANT_IDX via rf_score_model.py after any _PROFILES edit OR any
# constructive.exe rebuild (proxy ties can flip with ULP-level position changes).
#   tier-3 _M45_BAND_DROP: UNIVERSAL — mid cases are sum-bound even at 12 cores
#     (sum34/12 ~ 5.2s > max34 ~ 4.4s), so the cut pays on any machine.
#   tier-4 _M45_LOWCORE_DROP: applied only when detected cores <= _M45_CORES_MAX;
#     on high-core machines these bands are max-setter-bound (gain exactly 0, see
#     the 2026-07-02 3rd-tier verification) so the tier stays off = zero risk.
# M74 (2026-07-30): all four sets REGENERATED on audit_cache_ship.pkl, i.e. under
# the M71 knobs AND the shipping K=4/K=8 REFINE overlay. M67-F correction B
# (M67F_REPORT.md:249-258) had flagged that the old sets were fitted at REFINE=12
# and that the strict gate no longer held in the shipped configuration; on top of
# that the cache predated M71 entirely. Every band re-passes strict all-equal, and
# the mid cut got BIGGER (9 -> 15 profiles) at zero quality cost: local RF=1.0 is
# 1.2935 for shipped / +uni / +lowcore alike, while the mid-band wall @12c drops
# 2.21 -> 1.44s (-35%, vs -20% for the old 9-profile set).
# Derived on the SECOND ship cache, i.e. under the retuned mid K=6 (the first pass
# ran at K=8 and returned #5 where this one returns #10) -- the pool sets and the
# REFINE band are mutually dependent, so the chain was iterated to a fixpoint.
#
# M74 ALSO DEMOTED THIS TIER FROM UNIVERSAL TO CORES-GATED (_M45_MID_CORES_MAX
# below). The strict in-sample gate still holds exactly, but the M67-D/M55/M72
# doctrine says in-sample equality does not transfer -- and here it measurably
# does not: on the 80 held-out (60,100] cases, running the FULL mid pool scores
# -0.702% (30 better / 0 worse) against this cut. Meanwhile the cut buys almost
# nothing on a high-core grader: at 48 cores the mid band is max-setter-bound
# (c* max 15.2), its wall only moves 1.32 -> 1.30s, and rf_score_model projects
# +0.00% there. Paying 0.7% out-of-sample quality for a +0.00% wall is exactly
# the mistake tier-5 was introduced to undo on the M42 layer.
_M45_BAND_DROP: Tuple[Tuple[int, int, frozenset], ...] = (
    # mid cases: 15 profiles win nothing in n=61..100 (40 cases, kept 20).
    # Sum-bound (so worth cutting) only BELOW _M45_MID_CORES_MAX.
    (60, 100, frozenset({1, 6, 8, 10, 12, 14, 18, 21, 23, 24, 28, 30, 31, 32, 33})),
)
_M45_LOWCORE_DROP: Tuple[Tuple[int, int, frozenset], ...] = (
    # small band: deep floor at 12c (gain ~0) but sum-bound and scoring at low
    # cores; drops 24/35, kept 11 (strict-equal over its 20 cases).
    # Band wall @4c 4.57 -> 1.69s.
    (40, 60, frozenset({1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17,
                        20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 40})),
    # (100,110]: kept 8 — diversity floor + high weight keep this tier-4
    # (conservative); band wall @4c 4.70 -> 2.89s.
    (100, 110, frozenset({3, 6, 19, 23, 26})),
    # (110,inf): kept 7 — only true non-winners go; band wall @4c 5.37 -> 2.83s.
    (110, 10**9, frozenset({1, 2, 17, 18, 20, 27})),
)
# tier-4 activation: at 8 detected cores the tier still adds ~-2.2% real @ M=11
# (some walls are sum-bound there) and the strict gate makes it weakly winning at
# ANY cores; 8 also covers 8-vCPU/4-physical cloud boxes where the true gain is
# larger. Above 8 the increment (~-0.9% @12c) is not worth the kept-8 diversity
# squeeze on the high-weight (100,110] band.
_M45_CORES_MAX = 8

# M74 tier-3 activation (NEW): the mid-band cut fires only at or below this many
# detected cores. Threshold from the measured crossover c* = sum_i/max_i on the
# (60,100] band: p50 13.4 / MAX 15.2, i.e. from ~16 cores up every mid case is
# max-setter-bound and removing cheap profiles cannot shorten the wall (measured:
# 1.32 -> 1.30s at 48c, +0.00% projected). Below it the band is sum-bound and the
# cut is worth real time (@12c 2.21 -> 1.44s, @8c 3.32 -> 2.15s, @4c 6.64 -> 4.29s).
# 16 rather than 15.2 exactly because detected cores OVER-estimate effective ones
# (this 16-logical box has ~10 effective), so a box that reports 16 is still
# sum-bound and should keep cutting. Direction of mis-detection: _effective_cores()
# maps unknown -> 9999, so an undetectable box gets the tier OFF = the full pool =
# the quality-safe side, matching tier-4's fail-open convention.
_M45_MID_CORES_MAX = 16

# M67-F tier-5 (2026-07-26): the MIRROR of tier-4 — at HIGH core counts the M42
# big-redundant cut stops buying wall and only costs quality. M67-E measured that
# at 48 cores (the Beta box) the wall is the max-setter on 100/100 cases (sum/48
# is only 3-27% of it) and EVERY profile _BIG_REDUNDANT_IDX drops is cheaper than
# that max-setter -> restoring them buys the grader ~nothing in wall. M67-F Phase 1
# then measured, on 80 held-out (never-tuned) cases, that those cuts own
# theta_pool = 0.7636 of the +2.825% out-of-sample quality tax (44 better /
# 1 worse, best case -10.65%). Net projection with the 48c wall cost already
# charged (m67e_rf48.py `restoreIdx`, which does pay +5.7~8.7% wall on the heavy
# bands): official score -1.30% (s=1) / -0.55% (1.5) / -0.26% (2~2.5) — same sign
# at every machine speed. Threshold 32 rather than 48 so a 32-40 core grader also
# benefits; below it the 12c regime is sum-bound and the cut still pays for itself.
# 2026-07-27 CALIBRATION (raised 32 -> 40): the restored pool's c* = sum(dt)/max(dt)
# — the core count at which it stops being sum-bound — is min 15.5 / median 19.3 /
# MAX 22.5 over the 20 n>100 cases. At or above ~24 cores every heavy case is
# max-bound and the cost saturates at +6.0% (flat through 32/40/48/64), which is
# what the projection already charges. Below c* the wall grows like sum/cores and
# gets expensive fast: measured on this 16-LOGICAL box, restoring doubled the n>100
# wall (+97%, worst case +113%) because its EFFECTIVE parallelism is only ~10
# (12 physical, 8 of them slow E-cores; the model reproduces the measurement at
# ceff=10). Detected cores are therefore an UPPER bound on effective cores, so the
# threshold needs headroom over 22.5: at 32 detected a box with this one's 0.63
# effective ratio would land at ~20 < 22.5 and pay the sum-bound penalty. 40 keeps
# ~1.8x margin, still fires on the 48-core Beta box, and costs nothing there.
# ONLY the M42 layer is restored: mid-band tier-3 is ship-RED (recovery +0.620%
# < wall cost +0.695%, and the mid band is 0/40 at the RF floor so it pays in
# full). In-sample this tier is a NO-OP (M67-F Gate B: heavy band 20/20 cost-equal
# under the shipping K=4 overlay) — the whole gain is out-of-sample.
# ICCAD_M67F_TIER5=0 disables it.
_M67F_CORES_MIN = 40

# M49 (2026-07-07): band-gated REFINE truncation — the first MEASURED quality-
# vs-runtime trade (M41's was inferred). REFINE is 12 of every frame's 13 packs;
# m49_refine_probe.py (trace + variant, 20 n>100 cases x kept pool) showed:
#   - no exact cut exists (17/260 winning frames improve at the LAST pass), but
#   - truncating to 4 passes leaves 19/20 case costs BIT-IDENTICAL; the only
#     mover is case 85 (+0.95%, its winner improves at pass 11), weighted local
#     +0.027%, and all 20 cases stay median-independent WINs;
#   - band wall @12c 64.9->33.6s (-48%); projected real delta vs shipped:
#     @12c M=6/8/11 = -12.6%/-7.9%/-2.4%, M=14 -0.02%, worst case (RF floor /
#     RF ignored) +0.03%; low-core (4c) keeps gains at every median (sum-bound).
#   - K=4 weakly dominates K=6/8 everywhere measured at identical local cost.
# Applied as an env overlay on every profile run of a band case (the C++ knob
# ICCAD_REFINE_ITERS already exists); ICCAD_ADAPTIVE_REFINE=0 disables, and
# ICCAD_ADAPTIVE_POOL=0 (full quality-best pool) also restores full REFINE.
#
# M50 (2026-07-09): mid-band tier, measured the same way (m49_refine_probe.py
# trace/variant mid: 40 cases 60<n<=100 x kept 25):
#   - trace: no exact cut for mid either (40/1000 winning frames improve at
#     pass 11); refine cycles are frequent (538 events vs big's 2) but a cycle
#     fast-forward needs C++ changes for a mostly RF-floored band -> dropped.
#   - variant K=4 UNGATED fails the mid bar: 11 movers, weighted local +0.069%,
#     floor-saturated cells turn +0.06~0.07% at >=12c M>=11 (net loss on
#     high-core machines).
#   - variant K=8 PASSES: 6 movers (49/54/61/64/68/70), weighted local +0.028%,
#     band wall @12c 91.6->73.1s (-20.2%); projected real @12c M=6/8/11 =
#     -1.14/-0.85/-0.13%, worst cell +0.03% (the ceiling M49 accepted).
#   - K=4 as a LOW-CORE tier (detected cores <= _M45_CORES_MAX, fail-open, M45
#     tier-4 doctrine) dominates K=8 there: @4c -2.0~-3.5% at EVERY M, @8c wins
#     for M<=14 (worst @8c M=20 +0.05%); band wall -45%, sum -46% (sum-bound).
# Ship = two tiers: universal mid K=8 below, low-core mid K=4 override next.
# M74 (2026-07-30) re-swept K in [4,12] on both bands under M71 + the regenerated
# pool (m49_refine_probe.py variant 4,6,8,10 big|mid). Two changes of substance:
#   - big K=4 is now a quality WIN, not a quality trade: weighted local delta
#     -0.056% (movers 87 and 94, BOTH better; M49 measured +0.027% with case 85
#     the only mover). Band wall @12c -52.0%, 20/20 median-independent WINs, and
#     every projected cell over M in [6,20] x cores in {4,8,12,16} gains. M71
#     changed the sign: truncating REFINE now helps the heavy band.
#   - mid 8 -> 6. K=6 WEAKLY DOMINATES K=8 at every projected cell (e.g. @4c M=6
#     -2.13% vs -1.63%, @12c M=11 -0.10% both) at the same local quality
#     (+0.019% vs +0.018%, a 1e-6 difference) because it cuts more wall
#     (-31.0% vs -25.0% @12c). Worst cell +0.02%, inside the +0.03% M50 bar.
#     K=4 ungated still fails the mid bar (+0.049% local, worst cell +0.05%) and
#     so stays the low-core tier below.
_M49_REFINE_BAND: Tuple[Tuple[int, int, str], ...] = (
    (60, 100, "6"),                              # M50 universal tier (M74: 8 -> 6)
    (100, 10**9, "4"),                           # M49
)
# M50 low-core tier: replaces the universal band value when _effective_cores()
# <= _M45_CORES_MAX; mis-detection direction (unknown -> 9999) falls back to
# the universal tier = the safe profile.
_M50_REFINE_LOWCORE: Tuple[Tuple[int, int, str], ...] = (
    (60, 100, "4"),
)


def _band_env(block_count: int) -> Dict[str, str]:
    """M49/M50: per-case env overlay for every profile subprocess (band-gated
    REFINE truncation; M50 adds the cores-gated mid tier). Empty dict =
    pre-M49 behaviour."""
    if os.environ.get("ICCAD_ADAPTIVE_POOL", "1") == "0":
        return {}
    if os.environ.get("ICCAD_ADAPTIVE_REFINE", "1") == "0":
        return {}
    if _effective_cores() <= _M45_CORES_MAX:
        for lo, hi, iters in _M50_REFINE_LOWCORE:
            if lo < block_count <= hi:
                return {"ICCAD_REFINE_ITERS": iters}
    for lo, hi, iters in _M49_REFINE_BAND:
        if lo < block_count <= hi:
            return {"ICCAD_REFINE_ITERS": iters}
    return {}


# M71 (2026-07-29): cluster composite-item candidate enrichment, ported from the
# teammate's donor branch and re-verified here bit-for-bit (their local100 json is
# reproduced to the last digit by our own run). Two C++ knobs, both default-OFF in
# constructive.cpp so the binary alone stays bit-identical; the wrapper turns them
# on for every profile of every case:
#   ICCAD_CLUSTER_BND_EXPOSE    - in make_group_item(), rank the cluster's internal
#     candidate layouts by (boundary_bad, fragments, area, aspect) instead of
#     (fragments, boundary_bad, ...), AND add, for each candidate, a variant with
#     every boundary member pushed onto the item's own matching edge (kept only if
#     it stays overlap-free).
#   ICCAD_CLUSTER_BND_EDGE_PACK - add one more candidate layout: boundary members
#     laid around the item's rim, interior members filling the middle.
# This is the axis M33/M34/M37/M39 never touched: those searched cluster member
# ASPECT, never the composite item's candidate SET or its ranking key. M63 had
# already located the target (case 89's four pure-movable cluster-g1 boundary
# violators, the largest single-case term of the T2 violation bound) - and the
# movers match: 91/84/76/73/85/89/65 all improve.
# Measured (official eval, this box): 1.326473104916827 -> 1.305389893450635
# (-1.5894% weighted, 100/100 feasible, 56 better / 17 worse / 27 identical) with
# avg runtime 1.748 -> 1.521s, i.e. quality AND RuntimeFactor both improve. The
# heavy bands get FASTER (ratio 0.92/0.84/0.67 on (60,100]/(100,110]/(110,inf]).
# ICCAD_M71=0 restores the pre-M71 shipped behaviour bit-for-bit.
_M71_ENV: Dict[str, str] = {"ICCAD_CLUSTER_BND_EXPOSE": "1",
                            "ICCAD_CLUSTER_BND_EDGE_PACK": "1"}


def _m71_env() -> Dict[str, str]:
    """Per-profile env overlay for the M71 cluster-item knobs (default ON).
    Deliberately independent of ICCAD_ADAPTIVE_POOL/ADAPTIVE_REFINE: this is a
    pure quality/runtime win, not part of the RuntimeFactor tier stack."""
    if os.environ.get("ICCAD_M71", "1") == "0":
        return {}
    return dict(_M71_ENV)


# L137 (2026-08-19): the GORDIAN hint as the GLOBAL overlay, which is the form
# that was measured. The TIER form above (_l137_active) stays default-OFF -- it
# was worse on both axes (commit d64abe0).
#
# CORES-GATED at the same >=40 as route A / the shape LP / tier-5 / M80, and for
# the same reason M76 recorded: the quality (+0.0889% OOS s1 240) and the wall
# (+0.46%) were BOTH measured at 48c with route A and the LP live, so that is the
# only shape the number describes. Below the gate the pool is sum-bound, where
# the hint's extra refine passes are NOT absorbed by a max-setter, and nothing
# has measured it there. _effective_cores_hi() maps unknown -> 0, so a detection
# failure falls back to L136 behaviour (fail-CLOSED, the M67-F doctrine).
#
# HINT_REFINE=4 is part of the recipe, not a separate knob: it caps the refine
# loop on hinted runs only, and it is what turns the arm from "quality up, wall
# up 1.76%" into "quality up, wall up 0.46%".
_L137_ENV: Dict[str, str] = {"ICCAD_HINT_MODE": "1",
                             "ICCAD_HINT_REFINE": "4"}


def _l137_env() -> Dict[str, str]:
    """Per-profile env overlay for the L137 GORDIAN hint (global form).

    ICCAD_HINT_MODE=0 forces it off; any other explicit ambient value forces it
    on and wins over the recipe, so the A/B tools (l137_oos_ab.py,
    l113_ship_gate.py --env) keep measuring what they ask for."""
    v = os.environ.get("ICCAD_HINT_MODE", "")
    if v == "0":
        return {}
    if v == "" and _effective_cores_hi() < _L137_CORES_MIN:
        return {}
    ov = dict(_L137_ENV)
    for k in ov:
        amb = os.environ.get(k, "")
        if amb != "":
            ov[k] = amb
    return ov


def _profile_env(i: int, block_count: int) -> Dict[str, str]:
    """The per-profile env overlay _solve_impl applies to pool index `i`, in the
    wrapper's precedence order (profile dict, then band, then M71).

    An escape-tier index (M76) deliberately does NOT receive the M71 knobs — that
    omission IS the mechanism, and it is the one thing about the tier that cannot
    be checked from _pool_indices() alone, so it lives in a function the gates can
    call instead of being inlined in solve()."""
    ov = dict(_band_env(block_count))
    if i not in _M73_IDX:
        ov.update(_m71_env())
        ov.update(_l137_env())          # L137 GORDIAN hint, cores-gated >= 40
    return ov


def _effective_cores() -> int:
    """Detected parallelism for tier-4 gating. Conservative: logical count
    over-estimates effective cores -> mis-detection direction is 'tier stays
    off' = bit-identical shipped behaviour. ICCAD_ADAPTIVE_CORES forces a value
    (<=0/garbage -> auto); unknown -> 9999 (tier-4 off)."""
    v = os.environ.get("ICCAD_ADAPTIVE_CORES", "")
    if v:
        try:
            c = int(v)
            if c > 0:
                return c
        except ValueError:
            pass
    try:
        if hasattr(os, "sched_getaffinity"):        # Linux: cgroup/affinity-aware
            return len(os.sched_getaffinity(0)) or 9999
        return os.cpu_count() or 9999
    except Exception:
        return 9999


def _effective_cores_hi() -> int:
    """Detected parallelism for the M67-F tier-5 gate. MUST NOT reuse
    _effective_cores(): that one maps 'unknown' to 9999 so tier-4 (fires at
    cores <= 8) stays OFF on mis-detection, but tier-5 fires at cores >= 32, so
    the same sentinel would turn it ON wherever detection fails. This variant
    maps unknown to 0 so BOTH tiers fail to the shipped pool. ICCAD_ADAPTIVE_CORES
    forces a value (shared with tier-4/M50; <=0/garbage -> auto)."""
    v = os.environ.get("ICCAD_ADAPTIVE_CORES", "")
    if v:
        try:
            c = int(v)
            if c > 0:
                return c
        except ValueError:
            pass
    try:
        if hasattr(os, "sched_getaffinity"):        # Linux: cgroup/affinity-aware
            return len(os.sched_getaffinity(0)) or 0
        return os.cpu_count() or 0
    except Exception:
        return 0


def _pool_indices(block_count: int) -> List[int]:
    """Kept _PROFILES indices for this case size under the adaptive-pool tiers
    (M41 swap / M42 big-redundant / M45 band + low-core). ICCAD_ADAPTIVE_POOL=0
    returns the full pool."""
    # M72 (2026-07-30): call-time gate for the _M55_EXTRA tier. Read BEFORE the
    # ADAPTIVE_POOL=0 early return on purpose - the teammate's port checks it only
    # inside the loop below, so their `full` path leaks the four M72 profiles into
    # every ADAPTIVE_POOL=0 run (the offline quality anchor, the M53 L1/L3 modes and
    # this probe's own `full` endpoint), silently changing what those measure.
    m55 = os.environ.get("ICCAD_M55_POOL", "0") == "1"
    # M76: same call-time-and-before-the-early-return discipline for the escape
    # tier. ICCAD_M73_MIN_N band-gates it (0 = every band, 100 = heavy only), which
    # is the variant the teammate's own wall analysis pointed at: their all-band
    # RED came from a 12-core box where the mid band is sum-bound, and at 48 cores
    # the mid band is max-setter-bound (M74's c* max 15.2 is why tier-3 is now
    # cores-gated). One code path, two values, so both are measurable as arms.
    esc = _m73_active()
    if esc and block_count <= int(os.environ.get("ICCAD_M73_MIN_N", "0") or 0):
        esc = frozenset()
    # M80: same call-time-and-before-the-early-return discipline. Its own cores
    # gate lives inside _m80_active() because, unlike M72/M76, this tier is meant
    # to SHIP and the gate is the mechanism, not a measurement switch.
    extra = ((_M55_IDX if m55 else frozenset()) | esc
             | _m80_active(block_count) | _m124_active(block_count)
             | _l137_active(block_count))          # L137 GORDIAN-hint tier
    full = [i for i in range(len(_PROFILES))
            if i < _M55_BASE_LEN or i in extra]
    if os.environ.get("ICCAD_ADAPTIVE_POOL", "1") == "0":
        return full
    # M67-F (2026-07-22): OFFLINE-ONLY measurement knob, default 0 => this
    # function is bit-identical to shipped (same convention as ICCAD_L1_POOL).
    # NEVER set it in the submission. Why it exists: M67-E measured that at 48
    # cores (the Beta box) the wall is the max-setter on 100/100 cases and every
    # profile M42/M45 drops is CHEAPER than that max-setter, so restoring them
    # costs +0.00% wall there -- while still paying the M67-D out-of-sample
    # quality tax (+2.825% on n>100). =1 skips exactly those two layers so
    # m67_oos_probe.py `restore` can measure theta = the share of that OOS tax
    # owned by the pool cuts (break-even theta* = 0, upper bound -2.11%).
    # Deliberately NOT touched: the M41 swap filter and the M49/M50 REFINE band
    # (_band_env) -- both are the real 48c RF levers (+53%/+15% if dropped) --
    # and tier-4 _M45_LOWCORE_DROP, which only fires at <=8 detected cores and
    # is therefore fail-open in both the local 12c and the Beta 48c regime.
    restore = os.environ.get("ICCAD_M67F_RESTORE", "0") == "1"
    # M67-F tier-5 (SHIPPED, cores-gated — see _M67F_CORES_MIN): the same pool
    # effect as the offline restore knob, but ONLY the M42 layer and ONLY on a
    # >=32-core box, where M67-E proved those profiles are all cheaper than the
    # max-setter that actually sets the wall. Fails closed (unknown cores -> 0).
    tier5 = (os.environ.get("ICCAD_M67F_TIER5", "1") != "0"
             and _effective_cores_hi() >= _M67F_CORES_MIN)
    n_swap = int(os.environ.get("ICCAD_ADAPTIVE_N", "0"))
    n_free = int(os.environ.get("ICCAD_ADAPTIVE_FREE_N", "100"))
    drop_band: frozenset = frozenset()
    # M74: tier-3 is cores-gated now (see _M45_MID_CORES_MAX). Above the
    # threshold the mid band is max-setter-bound, so the cut buys ~0 wall while
    # still costing out-of-sample quality (-0.702% on 80 held-out mid cases).
    if (not restore and os.environ.get("ICCAD_ADAPTIVE_BAND", "1") != "0"
            and _effective_cores() <= _M45_MID_CORES_MAX):
        for lo, hi, d in _M45_BAND_DROP:                         # M45 tier-3
            if lo < block_count <= hi:
                drop_band = d
                break
    drop_low: frozenset = frozenset()
    if _effective_cores() <= _M45_CORES_MAX:                     # M45 tier-4
        for lo, hi, d in _M45_LOWCORE_DROP:
            if lo < block_count <= hi:
                drop_low = d
                break
    kept = []
    for i, p in enumerate(_PROFILES):
        if i >= _M55_BASE_LEN and i not in extra:
            continue                             # M72 / M76 tiers, default off
        # M41 stays CONTENT-based, so it also filters an escape twin of a swap
        # profile — that filter is about subprocess cost, which a knob-off copy
        # pays just as much. The INDEX-based drops below (M42/M45) deliberately do
        # not reach the appended tiers: their frozensets are over 0.._M55_BASE_LEN-1
        # and were derived on the shipped pool, so an escape twin is never pruned
        # by them. Same behaviour as the teammate's tier.
        if block_count > n_swap and ("ICCAD_ORDER_SWAP" in p
                                     or "ICCAD_ORDER_MOVE" in p):
            continue
        if (not restore and not tier5 and block_count > n_free
                and i in _BIG_REDUNDANT_IDX):
            continue                                             # M42 (tier-5 keeps)
        if i in drop_band or i in drop_low:
            continue
        kept.append(i)
    return kept if kept else full                                # never-empty guard

_RH = 1.4  # relative weight of the hpwl term in the proxy. The proxy uses hmin
           # (min hpwl over profiles) as a stand-in for the unknown baseline hpwl
           # hbase; since we never beat baseline, hmin > hbase by ~hmin/hbase≈1.3-1.4,
           # so the raw proxy under-weights hpwl vs the true cost (which divides by
           # hbase). _RH≈1.4 compensates -> proxy selection matches the oracle ceiling.
           # Flat basin 1.3-1.6 all hit oracle (1.4349); 1.0 gave 1.4369. (M13 _rh_sweep)


_SMOKE_INP: Optional[str] = None


def _binary_runs() -> bool:
    """True iff _BIN executes a trivial 1-block case end-to-end (M48). Catches
    binaries that exist but cannot run on this machine — a foreign-platform
    .exe shipped in the package, a truncated file — which exists()+mtime alone
    would accept, silently dropping EVERY case to the SA fallback."""
    global _SMOKE_INP
    if _SMOKE_INP is None:
        _SMOKE_INP = _serialize_input(1, [1.0], None, None, None, None, None)
    try:
        r = subprocess.run([str(_BIN)], input=_SMOKE_INP, capture_output=True,
                           text=True, timeout=30.0)
        return r.returncode == 0 and len(_parse_output(r.stdout, 1)) == 1
    except Exception:
        return False


def _ensure_compiled() -> bool:
    global _BIN
    # M67-A bundled-binary-first: on a POSIX grader a prebuilt Linux binary
    # (bin/constructive_linux, produced by M67-C) skips the compile entirely.
    # Gated off Windows and off ICCAD_CONSTRUCTIVE_BIN (explicit override wins);
    # a bundled binary that fails the 1-block smoke falls through to the chain.
    if os.name != "nt" and not os.environ.get("ICCAD_CONSTRUCTIVE_BIN"):
        bundled = _DIR / "bin" / "constructive_linux"
        if bundled.exists():
            try:
                os.chmod(bundled, os.stat(bundled).st_mode | 0o111)
            except Exception:
                pass
            prev = _BIN
            _BIN = bundled
            if _binary_runs():
                return True
            _BIN = prev
    src = _DIR / "constructive.cpp"
    if not src.exists():
        return _BIN.exists() and _binary_runs()
    if (_BIN.exists() and _BIN.stat().st_mtime >= src.stat().st_mtime
            and _binary_runs()):
        return True
    # M48 compile chain: the first candidate is the exact command used through
    # M47 (identical local behaviour); the others only matter where it is
    # missing (e.g. a Linux grader). -O2 is retried last in case -O3 fails.
    # ICCAD_CXX forces a specific compiler to the front of each round. The msys
    # candidate is Windows-only (M67-A: no doomed absolute-path exec on POSIX).
    compilers = ["g++", "clang++", "c++"]
    if os.name == "nt":
        compilers.insert(0, r"C:\msys64\ucrt64\bin\g++.exe")
    if os.environ.get("ICCAD_CXX"):
        compilers.insert(0, os.environ["ICCAD_CXX"])
    for opt in ("-O3", "-O2"):
        for gpp in compilers:
            try:
                r = subprocess.run(
                    [gpp, opt, "-std=c++17", "-o", str(_BIN), str(src)],
                    capture_output=True, text=True, timeout=240,
                )
                if r.returncode == 0 and _binary_runs():
                    return True
                if r.returncode != 0:
                    print(f"[constructive] {gpp} {opt} failed:\n{r.stderr}",
                          file=sys.stderr)
            except Exception as e:
                print(f"[constructive] compile error with {gpp} {opt}: {e}",
                      file=sys.stderr)
    return _BIN.exists() and _binary_runs()


def _hpwl_b2b_fast(positions, b2b):
    """calculate_hpwl_b2b with the tensor rows pre-converted via tolist().
    BIT-IDENTICAL to the harness version: tolist == float(tensor scalar) is the
    same exact fp32->fp64 widening, edge order and accumulation order unchanged,
    all arithmetic on python floats as before. Per-row tensor indexing was the
    proxy hot spot (M47b: 300/300 old==new, x11.5)."""
    if b2b is None or len(b2b) == 0:
        return 0.0
    total_wl = 0.0
    np_ = len(positions)
    for r in b2b.tolist():
        if r[0] == -1:
            continue
        i, j, weight = int(r[0]), int(r[1]), r[2]
        if i < np_ and j < np_:
            x1 = positions[i][0] + positions[i][2] / 2
            y1 = positions[i][1] + positions[i][3] / 2
            x2 = positions[j][0] + positions[j][2] / 2
            y2 = positions[j][1] + positions[j][3] / 2
            total_wl += weight * (abs(x2 - x1) + abs(y2 - y1))
    return total_wl


def _hpwl_p2b_fast(positions, p2b, pins):
    """calculate_hpwl_p2b, tolist-converted like _hpwl_b2b_fast (bit-identical)."""
    if p2b is None or len(p2b) == 0:
        return 0.0
    total_wl = 0.0
    np_ = len(positions)
    pins_l = pins.tolist() if pins is not None else []
    for r in p2b.tolist():
        if r[0] == -1:
            continue
        pin_idx, block_idx, weight = int(r[0]), int(r[1]), r[2]
        if block_idx < np_ and pin_idx < len(pins_l):
            px, py = pins_l[pin_idx][0], pins_l[pin_idx][1]
            bx = positions[block_idx][0] + positions[block_idx][2] / 2
            by = positions[block_idx][1] + positions[block_idx][3] / 2
            total_wl += weight * (abs(px - bx) + abs(py - by))
    return total_wl


def _proxy_metrics(positions, area_targets, b2b, p2b, pins, constraints, n):
    """Baseline-free (area, hpwl, vrel), computed EXACTLY like the harness so the
    live selector matches the offline-validated proxy. The C++ emits its own vrel
    too, but its union-find grouping (1e-3 tol) disagrees with shapely on ~34% of
    cases; replicating the harness here recovers the oracle-level selection.
    M47b: scalar tensor reads replaced by one-shot tolist() (values, order and
    formulas unchanged -> bit-identical; gated 300/300 on all 100 cases)."""
    xmin = min(p[0] for p in positions); ymin = min(p[1] for p in positions)
    xmax = max(p[0] + p[2] for p in positions); ymax = max(p[1] + p[3] for p in positions)
    area = (xmax - xmin) * (ymax - ymin)
    hpwl = _hpwl_b2b_fast(positions, b2b) + _hpwl_p2b_fast(positions, p2b, pins)

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    vb = vg = vm = 0
    nsoft = 0
    if ncols > 4:
        bound_l = constraints[:n, 4].tolist()
        clust_l = constraints[:n, 3].tolist()
        mib_l = constraints[:n, 2].tolist()
        nsoft = sum(1 for b in bound_l if b != 0)
        eps = 1e-6
        for i in range(n):
            code = int(bound_l[i])
            if code == 0:
                continue
            bx, by, bw, bh = positions[i]
            ok = True
            if code & 1: ok = ok and abs(bx - xmin) < eps
            if code & 2: ok = ok and abs(bx + bw - xmax) < eps
            if code & 4: ok = ok and abs(by + bh - ymax) < eps
            if code & 8: ok = ok and abs(by - ymin) < eps
            if not ok:
                vb += 1
        ngrp = int(max(clust_l)) if clust_l else 0
        for g in range(1, ngrp + 1):
            idx = [i for i in range(n) if int(clust_l[i]) == g]
            nsoft += max(0, len(idx) - 1)
            if len(idx) > 1 and _SHAPELY:
                u = _unary_union([_box(positions[i][0], positions[i][1],
                                       positions[i][0] + positions[i][2],
                                       positions[i][1] + positions[i][3]) for i in idx])
                if u.geom_type == "MultiPolygon":
                    vg += len(u.geoms) - 1
        nmib = int(max(mib_l)) if mib_l else 0
        for g in range(1, nmib + 1):
            idx = [i for i in range(n) if int(mib_l[i]) == g]
            nsoft += max(0, len(idx) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in idx}
            vm += len(shapes) - 1
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return {"area": area, "hpwl": hpwl, "vrel": vrel}


# M53 L1: overridable per-profile timeout (default 120s = shipped behaviour).
# The L1 quality pool (~84 profiles) oversubscribes cores, so nominal-48s
# profiles can stretch past 120s wall and would be SILENTLY dropped.
try:
    _PROFILE_TIMEOUT = float(os.environ.get("ICCAD_PROFILE_TIMEOUT", "120"))
except ValueError:
    _PROFILE_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# L110: route A runs on ONE global work queue.
#
# L109 measured route A as a net 2.8x LOSS.  That was not route A's doing: the
# wrapper nested two pools.  _solve_impl opens one thread per profile, and
# every _run_profile_route_a used to open its OWN window-sized pool, so the box
# ran n_profiles * window subprocesses at once (35 profiles * W12 = 420 on a
# 32-core machine; the 13-profile corner still asked for 156).  The W sweep
# degrading monotonically (W=4 -> 2.35s, W=16 -> 3.42s) is the signature of
# that oversubscription, not a property of per-frame parallelism.
#
# Fix: every (profile, frame) task in the process shares one executor sized to
# the core count, so the queue -- not the per-profile window -- is the only
# concurrency limit.  The outer per-profile threads stay: with route A on they
# never run work themselves, they only block on results.  Per-profile in-flight
# stays small (ICCAD_ROUTE_A, default 4 = max_trials for n>=60) because once
# the queue schedules, dispatching more frames than sequential selection could
# ever consume is pure waste that crowds out other profiles.
#
# Selection is untouched: the winner is still replayed from the pre-post-
# processing FSEL score in index order (strict '<', ties to the earlier index),
# which is what makes route A bit-identical to the sequential run.
_ROUTE_A_POOL = None
_ROUTE_A_LOCK = threading.Lock()
# Live/peak concurrent frame subprocesses. The peak is the direct measurement
# of the thing this change fixes -- it must never exceed _route_a_cores().
_ROUTE_A_LIVE = 0
_ROUTE_A_PEAK = 0
# Total frame-subprocess CPU-seconds and task count. The work MULTIPLIER
# (this vs the sequential pool's total) is what decides route A: the queue can
# only convert idle cores into wall, so it wins iff headroom > multiplier.
_ROUTE_A_WORK = 0.0
_ROUTE_A_TASKS = 0
_ROUTE_A_FSEL_MISMATCH = 0
# Times route A yielded nothing and _run_profile degraded to the sequential
# path. Must be 0 in any shipping gate: non-zero means the resolved binary
# does not answer ICCAD_FORCE_FRAME_IDX/ICCAD_FRAME_REPORT.
_ROUTE_A_DEGRADED = 0

# L110 two-phase: run_pipeline runs compact_layout/hpwl_push/slide ONCE, on the
# selected frame (constructive.cpp:1867-1875) -- OUTSIDE the frame loop. Route A
# splits the loop across processes, so each task pays that whole tail on its own
# single-frame winner and 3/4 of them get thrown away. Measured: the tail is
# 69-88% of a frame task, and dropping it takes the work multiplier vs the
# sequential profile from 2.03x to 1.44x (8 profiles, uncontended, min of 3).
#
# It is safe to drop because FSEL is emitted at :1852, BEFORE the tail -> the
# selection key is bit-identical either way (asserted over every scanned index
# of 8 profiles, and re-checked at runtime into _ROUTE_A_FSEL_MISMATCH).
# So: scan every candidate frame cheaply, then re-run ONLY the winner in full.
# ICCAD_ROUTE_A_2PHASE=0 restores the one-phase behaviour for A/B.
_ROUTE_A_SCAN_ENV = {"ICCAD_NO_COMPACT": "1", "ICCAD_NO_PUSH": "1"}


def _route_a_cores() -> int:
    """Concurrency cap for the global frame queue.

    Deliberately NOT _effective_cores()/_effective_cores_hi(): both honour
    ICCAD_ADAPTIVE_CORES, which is a *forced* tier-gating value (the 48c
    corners force 48 on this 32-core box). Sizing the queue from a forced
    value would re-create the oversubscription this replaces.
    ICCAD_ROUTE_A_CORES overrides for experiments."""
    v = os.environ.get("ICCAD_ROUTE_A_CORES", "")
    if v:
        try:
            c = int(v)
            if c > 0:
                return c
        except ValueError:
            pass
    try:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0)) or 1
        return os.cpu_count() or 1
    except Exception:
        return 1


def _route_a_pool():
    """The single process-wide frame queue (created on first route-A use)."""
    global _ROUTE_A_POOL
    if _ROUTE_A_POOL is None:
        with _ROUTE_A_LOCK:
            if _ROUTE_A_POOL is None:
                _ROUTE_A_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_route_a_cores(), thread_name_prefix="routeA")
    return _ROUTE_A_POOL


def route_a_stats() -> Dict[str, int]:
    """Peak concurrent frame subprocesses since the last reset (verification
    hook: peak must be <= cores)."""
    with _ROUTE_A_LOCK:
        return {"live": _ROUTE_A_LIVE, "peak": _ROUTE_A_PEAK,
                "work": _ROUTE_A_WORK, "tasks": _ROUTE_A_TASKS,
                "fsel_mismatch": _ROUTE_A_FSEL_MISMATCH,
                "degraded": _ROUTE_A_DEGRADED,
                "cores": _route_a_cores()}


def route_a_reset_stats() -> None:
    global _ROUTE_A_LIVE, _ROUTE_A_PEAK, _ROUTE_A_WORK, _ROUTE_A_TASKS
    global _ROUTE_A_FSEL_MISMATCH, _ROUTE_A_DEGRADED
    with _ROUTE_A_LOCK:
        _ROUTE_A_LIVE = _ROUTE_A_PEAK = _ROUTE_A_TASKS = 0
        _ROUTE_A_FSEL_MISMATCH = _ROUTE_A_DEGRADED = 0
        _ROUTE_A_WORK = 0.0


def _run_profile_frame(args):
    global _ROUTE_A_LIVE, _ROUTE_A_PEAK, _ROUTE_A_WORK, _ROUTE_A_TASKS
    exe, inp, base_env, idx = args
    env = dict(base_env)
    env["ICCAD_FORCE_FRAME_IDX"] = str(idx)
    env["ICCAD_FRAME_REPORT"] = "1"
    with _ROUTE_A_LOCK:
        _ROUTE_A_LIVE += 1
        new_peak = _ROUTE_A_LIVE > _ROUTE_A_PEAK
        if new_peak:
            _ROUTE_A_PEAK = _ROUTE_A_LIVE
        peak = _ROUTE_A_PEAK
    if new_peak:
        # Peaks are monotone and capped by the pool, so this fires at most
        # `cores` times per process -- cheap enough to keep out of a mode flag.
        # Lets the official-eval subprocess report the concurrency it actually
        # reached, which is the measurement L110 is making.
        p = os.environ.get("ICCAD_ROUTE_A_STATS", "")
        if p:
            try:
                with open(p, "w") as fh:
                    fh.write(f"{peak} {_route_a_cores()}\n")
            except Exception:
                pass
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(exe)], input=inp, capture_output=True, text=True,
                           timeout=_PROFILE_TIMEOUT, env=env)
    finally:
        wall = time.perf_counter() - t0
        with _ROUTE_A_LOCK:
            _ROUTE_A_LIVE -= 1
            _ROUTE_A_WORK += wall
            _ROUTE_A_TASKS += 1
    n_frames = max_trials = None
    ok, sc = 0, None
    for line in r.stderr.splitlines():
        if line.startswith("FRAMES "):
            _, a, b = line.split()
            n_frames, max_trials = int(a), int(b)
        elif line.startswith("FSEL "):
            _, o, s = line.split()
            ok, sc = int(o), float(s)
    return {
        "idx": idx,
        "ok": ok,
        "sc": sc,
        "out": r.stdout,
        "wall": wall,
        "n_frames": n_frames,
        "max_trials": max_trials,
    }


def _run_profile_route_a(env: Dict[str, str], inp: str, inflight: int) -> Dict[str, object]:
    """One profile's frames, dispatched onto the shared global queue.

    Frames are submitted strictly in index order and every submitted frame is
    awaited, so `got` is always the contiguous prefix [0, nxt). Submission
    stops as soon as `succ` (successes seen so far, all of them inside that
    prefix) reaches max_trials -> the prefix is guaranteed to contain the
    max_trials-th success, which is exactly what the sequential replay rule
    below needs. Same guarantee the old batch loop gave, without waiting for a
    whole batch to drain before refilling.
    """
    global _ROUTE_A_FSEL_MISMATCH
    # _BIN, NOT a hardcoded sibling: that is the binary _ensure_compiled()
    # resolved (bundled bin/constructive_linux first, else the compile chain,
    # each accepted only after the 1-block smoke). A hardcoded name would not
    # exist in the submission package at all -- and because the cores gate
    # keeps route A off on a <40-core box, nothing local would notice.
    exe = Path(env.get("ICCAD_CONSTRUCTIVE_BIN", str(_BIN)))
    pool = _route_a_pool()
    # from `env`, not os.environ: same convention as ICCAD_ROUTE_A above, so a
    # caller can A/B the two phases per profile without touching the process.
    two_phase = env.get("ICCAD_ROUTE_A_2PHASE", "1") != "0"
    greedy = env.get("ICCAD_ROUTE_A_GREEDY", "0") != "0"
    scan_env = dict(env, **_ROUTE_A_SCAN_ENV) if two_phase else env
    got = {}
    succ = 0
    waves = 0
    n_frames = max_trials = None
    nxt = 0                      # next index to submit
    pend = {}                    # future -> idx
    t0 = time.perf_counter()

    def _room() -> bool:
        # n_frames/max_trials are unknown until the first result lands, so the
        # opening submissions are blind -- the same cheap bet L108 documented
        # (indices past the end cost only setup and report ok=0), except the
        # bet is now `inflight` frames, not a whole window.
        #
        # The two stopping rules trade WORK against LATENCY, and which one is
        # right depends on whether the box is work- or latency-bound:
        #   thrifty (default) -- stop at succ + len(pend) >= max_trials. A frame
        #     in flight might yet succeed, so going past that speculates on a
        #     success we may already have. Scans what sequential scans (measured:
        #     8 frames, vs 11 for greedy) but caps in-profile parallelism at
        #     max_trials, so the successful frames get discovered serially.
        #   greedy (ICCAD_ROUTE_A_GREEDY=1) -- stop at succ >= max_trials, so
        #     `inflight` can exceed max_trials and the whole candidate span runs
        #     at once. Costs wasted frames, buys latency.
        # Either way the replay guarantee above holds: got is the contiguous
        # prefix [0, nxt) and submission only stops once that prefix is known to
        # contain max_trials successes (if in-flight frames fail, the count
        # drops back and the loop resumes).
        head = succ if greedy else succ + len(pend)
        return (len(pend) < inflight
                # Only the OPENING burst may be blind. Both "unknown yet"
                # escapes below stay true forever against a binary that
                # never emits FRAMES (anything pre-L108), so without this
                # bound nxt climbs without end and route A HANGS instead of
                # failing -- measured: no result in 600s, on a 21-block case.
                and (n_frames is not None or nxt < inflight)
                and (n_frames is None or nxt < n_frames)
                and (max_trials is None or head < max_trials))

    while True:
        while _room():
            pend[pool.submit(_run_profile_frame, (exe, inp, scan_env, nxt))] = nxt
            nxt += 1
        if not pend:
            break
        done, _ = concurrent.futures.wait(
            pend, return_when=concurrent.futures.FIRST_COMPLETED)
        waves += 1
        for f in done:
            pend.pop(f, None)
            r = f.result()
            got[r["idx"]] = r
            succ += int(bool(r["ok"]))
            if r["n_frames"] is not None:
                n_frames, max_trials = r["n_frames"], r["max_trials"]
    span = time.perf_counter() - t0
    best_i = None
    best_sc = None
    trials = 0
    for r in sorted(got.values(), key=lambda item: item["idx"]):
        if not r["ok"]:
            continue
        trials += 1
        if best_sc is None or r["sc"] < best_sc:
            best_sc = r["sc"]
            best_i = r["idx"]
        if max_trials is not None and trials >= max_trials:
            break
    if best_i is None:
        # "answered" separates "no frame won" (benign: the sequential path
        # would shelf-fallback here, and dropping the profile is what the
        # 48c on-vs-off A/B was validated with) from "the binary never
        # emitted a FRAMES line at all", which means it predates L108 and
        # every profile of every case is about to die.
        return {"out": None, "answered": n_frames is not None}
    win = got[best_i]
    if two_phase:
        # Phase 2: the scan runs produced the right SCORES but deliberately
        # skipped the tail, so their stdout is not a finished layout. Re-run the
        # winning index once, in full, THROUGH THE QUEUE -- calling it inline
        # would run it on the caller's outer per-profile thread, i.e. outside
        # the concurrency cap, which is the exact oversubscription L110 removes
        # (measured: peak 46 subprocesses on a 32-slot queue).
        win = pool.submit(_run_profile_frame, (exe, inp, env, best_i)).result()
        span = time.perf_counter() - t0
        if not win["ok"] or win["sc"] != got[best_i]["sc"]:
            # Cannot happen: FSEL precedes compact/push. Counted, not acted on
            # -- the full re-run IS the correct output either way, and silently
            # dropping the profile would hide the bug instead of surfacing it.
            with _ROUTE_A_LOCK:
                _ROUTE_A_FSEL_MISMATCH += 1
    return {
        "ok": win["ok"],
        "sc": win["sc"],
        "out": win["out"],
        "span": span,
        "dispatched": len(got) + (1 if two_phase else 0),
        "scanned": len(got),
        "two_phase": two_phase,
        "waves": waves,
        "n_frames": got[best_i].get("n_frames"),
        "max_trials": got[best_i].get("max_trials"),
    }


_ROUTE_A_INFLIGHT = 4        # = max_trials for n>=60; see _run_profile_route_a
_ROUTE_A_CORES_MIN = 40


def _route_a_default() -> int:
    """Cores-gated default, same bet and same shape as _m80_active/tier-5.

    Route A only converts IDLE cores into wall, so on a small box it is pure
    overhead. Uses _effective_cores_hi() (unknown -> 0) so a detection failure
    falls back to the shipped one-subprocess-per-profile path in BOTH
    directions. ICCAD_ROUTE_A=0 still forces it off; any explicit value wins."""
    return _ROUTE_A_INFLIGHT if _effective_cores_hi() >= _ROUTE_A_CORES_MIN else 0


def _run_profile(env_over: Dict[str, str], inp: str, n: int):
    """Run one profile; return positions or None.

    ICCAD_ROUTE_A unset = _route_a_default(): the global frame queue on a
    >=_ROUTE_A_CORES_MIN box, one subprocess per profile (= shipped) below
    that, and shipped again if core detection fails. 0 forces it off; any
    other value is the per-profile in-flight cap on the queue (1 = the
    default cap 4). See _run_profile_route_a for why the value no longer
    controls concurrency."""
    global _ROUTE_A_DEGRADED
    env = dict(os.environ)
    env.update(env_over)
    _v = env.get("ICCAD_ROUTE_A", "")
    try:
        route_a = int(_v) if _v != "" else _route_a_default()
    except ValueError:
        route_a = 0
    try:
        # _ROUTE_A_DEGRADED != 0 latches "this binary does not answer
        # ICCAD_FRAME_REPORT" for the rest of the process. The binary cannot
        # change mid-run, and a pre-L108 one ignores ICCAD_FORCE_FRAME_IDX,
        # so every wasted frame task runs the WHOLE pipeline -- ~5x the work
        # per profile. Scope of the saving, honestly: _solve_impl starts all
        # profile threads at once, so within the FIRST case they all read 0
        # and all pay; the latch saves cases 2..N, not case 1. Read without
        # the lock for that reason -- the race is already lost by design.
        if route_a >= 1 and _ROUTE_A_DEGRADED == 0:
            ra = _run_profile_route_a(env, inp,
                                      _ROUTE_A_INFLIGHT if route_a == 1 else route_a)
            if ra.get("out") is not None:
                return _parse_output(ra["out"], n)
            if ra.get("answered", True):
                return None            # no frame won; shipped behaviour
            # The binary never emitted FRAMES: it predates L108, so route A
            # is about to lose EVERY profile of EVERY case and the run sinks
            # to the SA fallback at ~10.0 -- invisibly, because the cores
            # gate keeps route A off on any box smaller than the grader.
            # Degrade to the shipped sequential path and make the gate hear
            # about it (make_submission.py verify greps stderr for
            # "fallback"; l113_ship_gate.py does too).
            with _ROUTE_A_LOCK:
                _ROUTE_A_DEGRADED += 1
                first = _ROUTE_A_DEGRADED == 1
            if first:
                print("[constructive] route A binary answers no "
                      "ICCAD_FRAME_REPORT (pre-L108?); sequential fallback",
                      file=sys.stderr)
        r = subprocess.run([str(_BIN)], input=inp, capture_output=True,
                           text=True, timeout=_PROFILE_TIMEOUT, env=env)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return _parse_output(r.stdout, n)
    except Exception:
        return None


def _row_fallback(block_count, area_targets, constraints, target_positions):
    """Last-resort layout (M48): preplaced blocks exactly at their targets,
    everything else in a single row strictly above them. Overlap-free with
    exact areas and fixed dims -> hard-feasible (soft violations only), so a
    case where even the SA fallback raises scores the feasible cap instead of
    the evaluator's exception-path M_PENALTY. Pure python, never raises."""
    def _f(v, default):
        try:
            x = float(v)
            return x if math.isfinite(x) else default
        except Exception:
            return default

    pos = [None] * block_count
    x_row, y_row = 0.0, 0.0
    for i in range(block_count):        # preplaced first: they pin the free zone
        try:
            pp = constraints is not None and int(constraints[i][1]) != 0
        except Exception:
            pp = False
        if pp and target_positions is not None:
            try:
                tx = _f(target_positions[i][0], 0.0)
                ty = _f(target_positions[i][1], 0.0)
                tw = max(_f(target_positions[i][2], 1.0), 1e-9)
                th = max(_f(target_positions[i][3], 1.0), 1e-9)
            except Exception:
                tx = ty = 0.0
                tw = th = 1.0
            pos[i] = (tx, ty, tw, th)
            y_row = max(y_row, ty + th)
    y_row += 1.0
    for i in range(block_count):
        if pos[i] is not None:
            continue
        w = h = -1.0
        try:                            # fixed-shape blocks carry exact (w,h)
            if target_positions is not None:
                w = _f(target_positions[i][2], -1.0)
                h = _f(target_positions[i][3], -1.0)
        except Exception:
            w = h = -1.0
        if not (w > 0 and h > 0):
            a = 1.0
            try:
                if area_targets is not None:
                    a = _f(area_targets[i], 1.0)
            except Exception:
                a = 1.0
            w = h = math.sqrt(max(a, 1e-12))
        pos[i] = (x_row, y_row, w, h)
        x_row += w + 1.0
    return pos


try:                                     # L137: numpy only; no scipy dependency
    import numpy as _gh_np
except Exception:                        # absent -> the hint goes inert
    _gh_np = None

_HINT_DENSITY = float(os.environ.get("ICCAD_HINT_DENSITY", "0.80"))
_HINT_LEVELS = int(os.environ.get("ICCAD_HINT_LEVELS", "16"))


def _gordian_hint(n, at, b2b, p2b, pins, cons, tp):
    """L137: a globally optimised centre for every block, as a placement hint.

    WHY. `estimate_anchors()` in constructive.cpp can only anchor a block to
    neighbours that are ALREADY PLACED, and it runs once when only the PREPLACED
    blocks are down -- so a block with no preplaced neighbour and no pin gets no
    anchor at all. The C++ header has said so since M9: "the first blocks are
    placed blind to HPWL". hpwl_gap is worth +10.11% of the score and is the one
    term this project has never moved on the shipped path.

    WHAT. GORDIAN's alternation: solve the quadratic wirelength problem, cut the
    region area-balanced, re-solve with one centre-of-gravity equality per region,
    cut again. The equality is weak enough that blocks still move to shorten wire
    and strong enough that they cannot re-pile, so wirelength keeps a say at every
    level instead of only the first. L130 measured this as the first mechanism to
    move hpwl_gap (-13.8% on the L129 candidate); L134 then closed that candidate
    on RUNTIME, not quality -- the alternation was never the expensive part.

    Block-level on purpose. L129 ran it over rigid cluster UNITS, but the C++
    forms its own items and consumes anchors PER BLOCK, so the units would just be
    rebuilt on the other side. Cluster members are instead collapsed onto their
    shared centroid at the end, which is the property that mattered.

    Priced before wiring (L137 gate 0): 19.7 ms weighted, 0.467% of the per-case
    wall, worst case 0.63%. Never raises: on any failure the caller falls back to
    the no-hint path, which is the shipped behaviour.
    """
    if _gh_np is None or n <= 2:
        return None
    np = _gh_np
    area = np.array([max(float(at[i]), 1e-9) for i in range(n)])
    pre = np.array([int(cons[i][1]) != 0 for i in range(n)])
    code = [int(cons[i][4]) for i in range(n)]
    clus = [int(cons[i][3]) for i in range(n)]

    fx = np.zeros(n)
    fy = np.zeros(n)
    if tp is not None:
        for i in range(n):
            if pre[i]:
                fx[i] = float(tp[i][0]) + float(tp[i][2]) / 2.0
                fy[i] = float(tp[i][1]) + float(tp[i][3]) / 2.0

    L = np.zeros((n, n))
    bx = np.zeros(n)
    by = np.zeros(n)
    for e in b2b.tolist():
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if i < 0 or j < 0 or i >= n or j >= n or i == j or w <= 0:
            continue
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    px = py = 0.0
    plist = pins.tolist()
    if plist:
        px = sum(float(p[0]) for p in plist) / len(plist)
        py = sum(float(p[1]) for p in plist) / len(plist)
    for e in p2b.tolist():
        p, j, w = int(e[0]), int(e[1]), float(e[2])
        if j < 0 or j >= n or p < 0 or p >= len(plist) or w <= 0:
            continue
        L[j, j] += w
        bx[j] += w * float(plist[p][0])
        by[j] += w * float(plist[p][1])
    # an unconnected block would leave a singular row; pull it weakly to the pins
    for k in range(n):
        L[k, k] += 1e-6
        bx[k] += 1e-6 * px
        by[k] += 1e-6 * py

    free = ~pre
    if not free.any():
        return None
    Lf = L[np.ix_(free, free)]
    rx = bx[free] - L[np.ix_(free, pre)] @ fx[pre]
    ry = by[free] - L[np.ix_(free, pre)] @ fy[pre]
    fidx = np.flatnonzero(free)
    pos = {int(k): t for t, k in enumerate(fidx)}
    F = len(fidx)

    def solve(rows, ux, uy):
        """min x'Lx - 2b'x subject to the region centre-of-gravity rows."""
        if not rows:
            A = None
        else:
            A = np.array(rows)
        try:
            if A is None:
                sx = np.linalg.solve(Lf, rx)
                sy = np.linalg.solve(Lf, ry)
            else:
                R = A.shape[0]
                K = np.zeros((F + R, F + R))
                K[:F, :F] = Lf
                K[:F, F:] = A.T
                K[F:, :F] = A
                K[F:, F:] = -1e-12 * np.eye(R)
                rhs = np.zeros((F + R, 2))
                rhs[:F, 0] = rx
                rhs[:F, 1] = ry
                rhs[F:, 0] = ux
                rhs[F:, 1] = uy
                sol = np.linalg.solve(K, rhs)
                sx, sy = sol[:F, 0], sol[:F, 1]
        except Exception:
            return None, None
        ox, oy = fx.copy(), fy.copy()
        ox[free], oy[free] = sx, sy
        return ox, oy

    cx, cy = solve([], None, None)
    if cx is None:
        return None

    # 🚨 THE BOX MUST BE ANCHORED AT THE ORIGIN, not at the solve's own minimum.
    # constructive.cpp packs into a frame [0,fw] x [0,fh] -- its LEFT test is
    # literally `fabs(x - 0.0)` -- and preplaced blocks sit at their absolute
    # tx/ty while pins carry absolute coordinates, so the C++ coordinate space
    # starts at 0. An unconstrained quadratic solve does NOT: it floats wherever
    # the pins pull it. Anchoring the region box at min(cx), min(cy) (which is
    # what L129 did, correctly, because it placed into its own frame) produces
    # hint coordinates in a different origin from the consumer's, and the anchor
    # pull then drags every block toward a meaningless point.
    # MEASURED with the floating origin: 48c 1.2284738 -> 1.2344230, i.e. 0.48%
    # WORSE, 59/100 cases changed. The mechanism was never given a fair test.
    x0 = y0 = 0.0
    side = math.sqrt(float(area.sum()) / max(_HINT_DENSITY, 0.05))
    if pre.any():
        side = max(side, float(cx[pre].max()), float(cy[pre].max()))

    def rank(i, horiz):
        """A boundary block must end up at that extreme, and nothing in the
        quadratic objective knows it -- L130 measured the alternation buying hpwl
        and area and paying more than both back in boundary violations without
        this. Sorting it to the matching end of every cut it meets lands it in
        the outermost leaf."""
        lo, hi = (code[i] & 1, code[i] & 2) if horiz else (code[i] & 8, code[i] & 4)
        return 0 if (lo and not hi) else (2 if (hi and not lo) else 1)

    regions = [(list(range(n)), x0, y0, side, side)]
    for _lvl in range(max(1, _HINT_LEVELS)):
        nxt = []
        for idx, rx0, ry0, w, h in regions:
            if len(idx) <= 1:
                nxt.append((idx, rx0, ry0, w, h))
                continue
            horiz = w >= h
            key = cx if horiz else cy
            order = sorted(idx, key=lambda q: (rank(q, horiz), key[q], q))
            half, run, cut = float(area[order].sum()) / 2.0, 0.0, 1
            for t, q in enumerate(order):
                run += area[q]
                if run >= half:
                    cut = max(1, min(len(order) - 1, t + 1))
                    break
            lft, rgt = order[:cut], order[cut:]
            fr = float(area[lft].sum()) / max(float(area[order].sum()), 1e-9)
            if horiz:
                nxt.append((lft, rx0, ry0, w * fr, h))
                nxt.append((rgt, rx0 + w * fr, ry0, w * (1 - fr), h))
            else:
                nxt.append((lft, rx0, ry0, w, h * fr))
                nxt.append((rgt, rx0, ry0 + h * fr, w, h * (1 - fr)))
        if len(nxt) == len(regions):
            break
        regions = nxt
        rows, ux, uy = [], [], []
        for idx, rx0, ry0, w, h in regions:
            mem = [q for q in idx if free[q]]
            tot = float(sum(area[q] for q in mem))
            if not mem or tot <= 1e-9:
                continue
            r = np.zeros(F)
            for q in mem:
                r[pos[q]] = area[q] / tot
            rows.append(r)
            ux.append(rx0 + w / 2.0)
            uy.append(ry0 + h / 2.0)
        nx, ny = solve(rows, ux, uy)
        if nx is None:
            break
        cx, cy = nx, ny

    # cluster members share a centroid: L129 placed each cluster as one rigid
    # unit, and this is the part of that which the anchor consumer can use.
    groups = {}
    for i in range(n):
        if clus[i]:
            groups.setdefault(clus[i], []).append(i)
    for mem in groups.values():
        if len(mem) < 2:
            continue
        tot = float(sum(area[q] for q in mem))
        gx = float(sum(area[q] * cx[q] for q in mem) / tot)
        gy = float(sum(area[q] * cy[q] for q in mem) / tot)
        for q in mem:
            if not pre[q]:
                cx[q], cy[q] = gx, gy

    # 🚨 EMIT FRAME-RELATIVE, in [0,1]^2. constructive.cpp does not pack into one
    # frame -- it tries a whole set (scales 1.05-2.10 x several aspects) and keeps
    # the best. An absolute hint silently assumes ONE of them, and fights every
    # other candidate, including the tall/wide frames that win on many cases: the
    # box here is square by construction while e.g. case 54 packs into 141x219.
    # The consumer scales by (fw,fh) at the point of use, where the frame is known.
    # MEASURED absolute, after the origin was already fixed: 1.2311612 against a
    # 1.2284738 baseline -- still 0.219% worse, which is what sent me here.
    inv = 1.0 / max(side, 1e-9)
    return [(min(max(float(cx[i]) * inv, 0.0), 1.0),
             min(max(float(cy[i]) * inv, 0.0), 1.0)) for i in range(n)]


_SNAP_TOL = 1e-9


def _snap_group_abutment(pos, constraints, block_count, tol=_SNAP_TOL):
    """L131: close sub-ULP gaps between members of the same cluster group.

    🚨 `origin + offset` DOES NOT ABUT in doubles. The evaluator builds a block's
    far edge as `x + w` and floating-point addition is not associative, so
    `(o+ox)+w` and `o+(ox+w)` differ by an ULP: a packing that abuts exactly in
    exact arithmetic lands +-2.8e-14 off, and `unary_union` then either merges
    the group or splits it. A split costs `connected_components - 1` grouping
    violations.

    MEASURED on the shipped 48c result: 2 of its 16 grouping violations are this,
    on cases 69 (gap +2.842e-14) and 78 (+1.421e-14), both heavy. Snapping them
    and re-running the OFFICIAL evaluator moves the weighted total
    1.236791669773 -> 1.235854685125, i.e. **+0.0758%**, 0 new infeasibilities.
    See L131_REPORT.md.

    WHY A SNAP IS SAFE, and it is the reason this can be a post-process at all:
      * `check_overlap` (iccad2026_evaluate.py:223) ignores overlaps below
        **1e-6** on both axes -- "touching edges OK". Gaps here are ~1e-14 and a
        snap moves a block by at most `tol` = 1e-9, so it cannot manufacture an
        overlap violation, with five orders of magnitude to spare.
      * preplaced blocks are NEVER moved (position is a HARD constraint);
      * widths and heights are never touched (dimensions and area are HARD);
      * only gaps strictly inside (0, tol] are closed, so a real gap stays a real
        gap and a group that genuinely has two components keeps both.

    Assigning `x_j = x_i + w_i` makes the two edges the identical float -- the
    same expression the evaluator uses for block i's far edge -- so they touch.

    Never raises: this sits on the shipped return path and M48's rule is that
    nothing escapes `solve()`. On any surprise it returns the input unchanged.
    """
    try:
        n = int(block_count)
        if n <= 0 or constraints is None or pos is None or len(pos) < n:
            return pos
        groups = {}
        pre = [False] * n
        for i in range(n):
            row = constraints[i]
            pre[i] = int(row[1]) != 0
            g = int(row[3])
            if g:
                groups.setdefault(g, []).append(i)
        groups = [m for m in groups.values() if len(m) > 1]
        if not groups:
            return pos

        P = [list(map(float, q)) for q in pos]
        moved = False
        for mem in groups:
            for _ in range(4):                       # a chain a-b-c settles fast
                for ax in (0, 1):
                    oth = 1 - ax
                    order = sorted(mem, key=lambda i: P[i][ax])
                    for a in order:
                        for b in order:
                            if a == b or pre[b]:
                                continue
                            # a shared edge needs real overlap on the other axis
                            lo = max(P[a][oth], P[b][oth])
                            hi = min(P[a][oth] + P[a][oth + 2],
                                     P[b][oth] + P[b][oth + 2])
                            if hi - lo <= tol:
                                continue
                            far = P[a][ax] + P[a][ax + 2]
                            gap = P[b][ax] - far
                            if 0.0 < gap <= tol:
                                P[b][ax] = far
                                moved = True
        if not moved:
            return pos
        out = [tuple(q) for q in P]
        return out + list(pos[n:]) if len(pos) > n else out
    except Exception:
        return pos


class MyOptimizer(FloorplanOptimizer):
    """Constructive fixed-outline placer, portfolio + proxy selection."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._ok = _ensure_compiled()
        self._single = os.environ.get("ICCAD_CONSTRUCTIVE_SINGLE") == "1"
        if not self._ok:
            print("[constructive] binary unavailable; falling back to python SA",
                  file=sys.stderr)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        # M48 blanket safety net: an exception escaping solve() makes the
        # evaluator record M_PENALTY 10.0 for the case (iccad2026_evaluate.py
        # :917-922). Never triggered on the 100 local cases; a hidden weird
        # case degrades to the SA fallback (feasible 100/100, M43) and, if
        # even that raises, to a trivial hard-feasible row layout.
        # L131: every exit goes through the abutment snap. It is a no-op unless a
        # cluster group carries a sub-ULP gap, and it cannot fail (see the
        # function). Applied here rather than inside the placer so it covers the
        # SA and row fallbacks too -- they emit `origin + offset` layouts as well.
        def _snap(p):
            return _snap_group_abutment(p, constraints, block_count)
        # L157: the case clock. The per-case RF-floor gate needs this
        # case's OWN elapsed time and solve() is the only place that
        # knows when the case began. Stamped before the fallbacks too,
        # so a case that lands on SA carries a valid clock, not a stale
        # one from the previous case.
        _case_clock_start()
        try:
            return _snap(self._solve_impl(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions))
        except Exception as e:
            print(f"[constructive] solve raised {e!r}; python SA fallback",
                  file=sys.stderr)
        try:
            return _snap(python_sa_solve(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions))
        except Exception as e:
            print(f"[constructive] SA fallback raised {e!r}; row fallback",
                  file=sys.stderr)
            return _snap(_row_fallback(block_count, area_targets, constraints,
                                       target_positions))

    def _solve_impl(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        if not self._ok:
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        # L137: the GORDIAN hint. OFF unless ICCAD_HINT_MODE>0, and the binary
        # ignores the block unless the same knob is set on its side, so with the
        # knob unset this is bit-identical to L136. Any failure falls back to
        # gnn_hint=None, i.e. the shipped path -- the hint is an optimisation,
        # never a dependency.
        # The hint block rides in the ONE serialized input every profile shares,
        # so it is computed at most once per case. A profile without
        # ICCAD_HINT_MODE parses the block and ignores it, which is why the
        # untiered profiles stay bit-identical -- verified by the gate.
        # Computed when EITHER the tier is live (the deployment shape) or the
        # global knob is set (the A/B shape kept for measurement).
        _hint = None
        _want = (bool(_l137_env())                     # global form (shipped)
                 or bool(_l137_active(block_count)))   # tier form (default off)
        if _want:
            try:
                _hint = _gordian_hint(block_count, area_targets,
                                      b2b_connectivity, p2b_connectivity,
                                      pins_pos, constraints, target_positions)
            except Exception:
                _hint = None
        inp = _serialize_input(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions, gnn_hint=_hint,
        )
        profiles = _PROFILES[:1] if self._single else _PROFILES

        # M41/M42: RuntimeFactor lever (the one scoring term the local harness hides).
        # Official Cost multiplies EACH case by max(0.7,(runtime/median)^0.3)
        # (iccad2026_evaluate.py:552), but the local harness forces RF=1.0
        # (:924-940) -> the whole M1-M37 portfolio history is blind to runtime and
        # 1.3269 is the RF=1.0 fiction. cost ∝ t^0.3, so trimming the big-case wall
        # (n>=110 = 60% of weight) is a median-INDEPENDENT real-score gain whenever
        # the dropped profiles don't change the selected winner there. Two tiers,
        # both filtered by ORIGINAL _PROFILES index (robust to pool ordering):
        #  M41 (ICCAD_ADAPTIVE_N, default 0): drop the 6 ORDER_SWAP/ORDER_MOVE
        #    profiles (audit cpu max ~19s vs ~8s) — they set the 18-20s wall yet the
        #    proxy never selects them on big cases. -0.06% RF=1.0 local, projected
        #    real ~-12% @ M=11 (1.4742->1.2904), avg 9.89->5.90s.
        #  M42 (ICCAD_ADAPTIVE_FREE_N, default 100): ALSO drop the 22 _BIG_REDUNDANT_IDX
        #    build-time profiles for block_count>100 — they win NO n>100 case (the swap
        #    argument applied per-big-n), so dropping them is wall-only: RF=1.0 local
        #    stays 1.3277 BIT-IDENTICALLY while the n=120 wall halves 15.6->8.0s,
        #    projecting a FURTHER ~-11% real @ M=11 (1.2904->1.1473), all 20 n>100
        #    cases median-INDEPENDENT WINs, robust over median in [6,20]s.
        # Default ON; ICCAD_ADAPTIVE_POOL=0 restores the full 41-profile pool
        # (quality-best 1.3248, full REFINE). Set ICCAD_ADAPTIVE_FREE_N huge
        # (e.g. 9999) for M41-only behaviour.
        #  M45 (2026-07-02): two more tiers inside _pool_indices() — tier-3 band-
        #    scoped mid-case drops (UNIVERSAL, ICCAD_ADAPTIVE_BAND=0 disables) and
        #    tier-4 low-core drops (only when _effective_cores() <= _M45_CORES_MAX;
        #    ICCAD_ADAPTIVE_CORES forces/disables detection). Both under the strict
        #    selection-preserving gate -> local RF=1.0 score unchanged (1.3277).
        #  M49 (2026-07-07): band-gated REFINE truncation via _band_env() — every
        #    profile of an n>100 case runs with ICCAD_REFINE_ITERS=4 (measured:
        #    19/20 case costs bit-identical, +0.027% local, band wall -48%;
        #    ICCAD_ADAPTIVE_REFINE=0 disables). See _M49_REFINE_BAND.
        #  M71 (2026-07-29): the cluster composite-item knobs ride the same
        #    per-profile overlay (see _m71_env); unlike the M49/M50 band they
        #    apply to every case size and are independent of the adaptive tiers.
        #  M76 (2026-08-01): ...except the escape tier, which deliberately runs
        #    knob-OFF so the pool carries both variants of a host and the proxy
        #    arbitrates per case. Skipping the M71 update on those indices IS the
        #    mechanism — without it the tier would be a pure duplicate of its host
        #    and cost wall for nothing.
        if not self._single:
            profiles = []
            for i in _pool_indices(block_count):
                ov = _profile_env(i, block_count)
                profiles.append(dict(_PROFILES[i], **ov) if ov else _PROFILES[i])

        # M47: compute each profile's proxy on the MAIN thread as soon as that
        # profile finishes (as_completed), overlapping the still-running slower
        # profiles. The serial post-pool proxy tail was 29% of the scored wall
        # on n>100 (2.9s on n=120); running the 13 proxies CONCURRENTLY in the
        # worker threads was 4x WORSE (pure-Python GIL thrash), so exactly one
        # proxy runs at a time. Same _proxy_metrics inputs, results stored by
        # profile index -> same cands order and values -> argmin unchanged. A
        # proxy failure degrades to (pos, None) and is recomputed below on the
        # original exception path (a lone candidate never reads its proxy).
        margs = (area_targets, b2b_connectivity, p2b_connectivity,
                 pins_pos, constraints, block_count)
        if len(profiles) == 1:
            results = [(_run_profile(profiles[0], inp, block_count), None)]
        else:
            results = [(None, None)] * len(profiles)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as ex:
                futs = {ex.submit(_run_profile, p, inp, block_count): i
                        for i, p in enumerate(profiles)}
                for f in concurrent.futures.as_completed(futs):
                    pos = f.result()
                    m = None
                    if pos is not None:
                        try:
                            m = _proxy_metrics(pos, *margs)
                        except Exception:
                            m = None
                    results[futs[f]] = (pos, m)

        kept = [(pos, m) for pos, m in results if pos is not None]
        cands = [pos for pos, _ in kept]
        if not cands:
            print("[constructive] all profiles failed; python SA fallback",
                  file=sys.stderr)
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        if len(cands) == 1:
            return _shape_lp_maybe(cands[0], block_count, area_targets,
                                   b2b_connectivity, p2b_connectivity,
                                   pins_pos, constraints, margs)

        # Baseline-free proxy selection: cost ~ (area/A + hpwl/H)*exp(2*vrel).
        # Metrics normally arrive precomputed from the profile threads (M47).
        # M48: if the proxy/selection stage itself raises (all thread-side
        # proxies already failed on some weird hidden case), any C++ candidate
        # is hard-feasible and beats the SA fallback by miles -> cands[0].
        try:
            metrics = [m if m is not None else
                       _proxy_metrics(pos, *margs)
                       for pos, m in kept]
            sumA = sum(max(0.0, float(area_targets[i])) for i in range(block_count))
            A_hat = 1.035 * max(sumA, 1e-9)
            hmin = min(m["hpwl"] for m in metrics) or 1.0
            best_pos, best_proxy = cands[0], float("inf")
            for pos, m in zip(cands, metrics):
                proxy = (m["area"] / A_hat + _RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                if proxy < best_proxy:
                    best_proxy, best_pos = proxy, pos
            return _shape_lp_maybe(best_pos, block_count, area_targets,
                                   b2b_connectivity, p2b_connectivity,
                                   pins_pos, constraints, margs)
        except Exception as e:
            print(f"[constructive] proxy selection raised {e!r}; keeping first "
                  f"candidate", file=sys.stderr)
            return cands[0]


# ═════════════════════════════════════════════════════════════════════════════
# L114: shape LP post-processing (OFF unless ICCAD_SHAPE_LP=1).
#
# Re-chooses every reshapeable block's aspect at FIXED topology by putting shape
# variables into the constraint-graph LP that already enforces preplaced pinning,
# boundary touch, cluster contiguity and exact-linear HPWL. Measured offline
# against the M80 @48c anchor: +2.3559% in-set at k=1, +2.4671% out-of-sample
# (104.7% transfer, 0 regressions, 0 infeasible on both corpora).
#
# ⚠️ THIS IS NOT A BIT-IDENTICAL CHANGE. The LP is massively degenerate, so a
# reformulated matrix can return a different optimal vertex with the same
# objective. It is validated by official eval + OOS, NOT by an equivalence gate.
# The HPWL pruning that makes it fast is exact on 99/100 in-set and 99/100 OOS
# cases; on the odd one the sign verification lets a violating vertex through
# (in-set that case got better by 2.9e-3, out-of-sample another got worse by
# 1.6e-2, net weighted +0.0008%). What actually protects the output is the
# three-way keep gate below: proxy must improve and not regress, hard_ok must
# pass, and the official feasibility is unchanged.
#
# Extracted verbatim from teammate_m71_screen/l100_lp_speed.py (accelerated
# builder + HPWL pruning + vectorised guard) and m53_l3_probe.py (comp_split),
# so the shipped code and the measured code are the same code. `l3` below is a
# shim carrying exactly the two attributes that were used.
# ═════════════════════════════════════════════════════════════════════════════
try:
    from collections import Counter          # used by the extracted builder
    import numpy as _lp_np
    from scipy import sparse as _lp_sparse
    from scipy.optimize import linprog as _lp_linprog
    _LP_IMPORTS_OK = True
except Exception:                       # scipy absent -> the knob is inert
    _LP_IMPORTS_OK = False
_LP_IMPORTS_OK = _LP_IMPORTS_OK and _SHAPELY

if _LP_IMPORTS_OK:
    np = _lp_np
    sparse = _lp_sparse
    linprog = _lp_linprog
    # comp_split below is verbatim from m53_l3_probe, which imported these two
    # shapely names under different aliases; bind them rather than edit it.
    _sbox = _box
    unary_union = _unary_union


class _LPShim:
    """Stands in for m53_l3_probe: the extracted code touches CASES and
    comp_split and nothing else."""
    CASES: Dict[object, dict] = {}

    @staticmethod
    def comp_split(P, mem):
        return comp_split(P, mem)


l3 = _LPShim


def comp_split(P, mem):
    """Cluster members -> connected components exactly as the evaluator sees
    them (unary_union geoms; zero tolerance)."""
    boxes = {i: _sbox(P[i][0], P[i][1], P[i][0] + P[i][2], P[i][1] + P[i][3])
             for i in mem}
    u = unary_union(list(boxes.values()))
    geoms = [u] if u.geom_type == "Polygon" else list(u.geoms)
    comps = [[] for _ in geoms]
    for i in mem:
        b = boxes[i]
        k = max(range(len(geoms)), key=lambda t: geoms[t].intersection(b).area)
        comps[k].append(i)
    return comps


EPS_BND = 1e-6


EPS_OVL = 1e-6


_LP_AREA_TOL = 0.008


def decompose(ci, P):
    c = l3.CASES[ci]
    n, cn = c["n"], c["cn"]
    frozen_blk = {i for i in range(n) if cn[i][1] != 0}
    unit_of = [None] * n
    units = []
    group_units, group_comp0 = {}, {}
    for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
        mem = [i for i in range(n) if cn[i][3] == g]
        comps = l3.comp_split(P, mem)
        group_comp0[g] = len(comps)
        gset = set()
        for cm in comps:
            if any(i in frozen_blk for i in cm):
                frozen_blk.update(cm)
            else:
                uid = len(units)
                units.append(cm)
                for i in cm:
                    unit_of[i] = uid
                gset.add(uid)
        group_units[g] = gset
    for i in range(n):
        if unit_of[i] is None and i not in frozen_blk:
            unit_of[i] = len(units)
            units.append([i])
    return units, unit_of, group_units, group_comp0


def reshapeable(ci, units):
    cn = l3.CASES[ci]["cn"]
    out = {}
    for uid, mem in enumerate(units):
        if len(mem) != 1:
            continue
        i = mem[0]
        if cn[i][0] != 0 or cn[i][1] != 0 or cn[i][2] > 0:
            continue
        code = cn[i][4]
        if (code & 1 and code & 2) or (code & 4 and code & 8):
            continue
        out[uid] = i
    return out


def _aggregate_pairwise_edges(c, unit_of):
    b2l = []
    b2l_agg = {}
    for i, j, w in c["b2l"]:
        if w <= 0.0:
            b2l.append((i, j, w))
            continue

        ui, uj = unit_of[i], unit_of[j]
        if ui == uj or ui is None or uj is None:
            b2l.append((i, j, w))
            continue

        key = (min(i, j), max(i, j))
        if key not in b2l_agg:
            b2l_agg[key] = (i, j, w)
        else:
            oi, oj, ow = b2l_agg[key]
            b2l_agg[key] = (oi, oj, ow + w)
    b2l.extend((i, j, w) for i, j, w in b2l_agg.values())

    p2l = []
    p2l_agg = {}
    for p, i, w in c["p2l"]:
        if w <= 0.0:
            p2l.append((p, i, w))
            continue

        ui = unit_of[i]
        if ui is None:
            p2l.append((p, i, w))
            continue

        px, py = c["pin"][p]
        key = (i, float(px), float(py))
        if key not in p2l_agg:
            p2l_agg[key] = [p, i, w]
        else:
            p2l_agg[key][2] += w
    p2l.extend((p, i, w) for p, i, w in p2l_agg.values())
    return b2l, p2l


def _sep_reduction_mask(rows, n, P, unit_of, sv, resh, rho):
    """Transitive reduction of the same-axis separation rows. EXACT.

    A row for the ordered block pair (a, c) on the x axis reads

        d_u(a) + dsize_u(a) - d_u(c)  <=  gap_ac ,   gap_ac = x_c - (x_a + w_a)

    Adding the rows along a path a -> b -> c on the same axis and substituting
    the anchor identity `gap_ab + gap_bc = gap_ac - w_b` telescopes to

        d_u(a) + dsize_u(a) - d_u(c)  <=  gap_ac - (w_b + dsize_u(b))

    which implies the direct row as soon as `w_b + dsize_u(b) >= 0`, and the
    same collapse holds for any path length and for the y axis.  Only
    single-block units are reshapeable (`reshapeable()` skips `len(mem) != 1`),
    so that quantity is `(1 - rho) * w_b > 0` when b is reshapeable and `w_b > 0`
    otherwise -- every block qualifies as an intermediate today.  The test is
    still written out per block, so the reduction stays sound by construction if
    reshaping is ever widened to rigid multi-block units.

    The graph is per BLOCK and must stay that way.  A rigid cluster maps many
    blocks onto one unit, so a unit-level chain U1 -> U3 -> U2 is in general
    built from rows about *different* block pairs, and the identity above does
    not hold across them.  Reducing at unit level is what made the first cut of
    this drop non-redundant rows and move the LP objective by 12%.

    Removals are computed against the original edge set and applied in one shot.
    That is safe: take a longest path between the endpoints of a removed row
    (finite, since a same-axis edge forces `x_a < x_b`, so the graph is a DAG);
    none of its edges can itself be removable, or splicing in its replacement
    would yield a longer path.  So every removed row stays implied by rows that
    survive.
    """
    keep = [True] * len(rows)
    for axis in (0, 1):
        idxs = [k for k, r in enumerate(rows) if r["axis"] == axis]
        if len(idxs) <= 1:
            continue
        size = [P[b][2 + axis] for b in range(n)]
        usable = [False] * n
        for b in range(n):
            u = unit_of[b]
            lb = -rho * size[b] if (u is not None and u in sv) else 0.0
            usable[b] = size[b] + lb >= 0.0

        # chain only through non-negative gaps, which is what makes it a DAG
        adj = [0] * n
        for k in idxs:
            if rows[k]["rhs"] >= 0.0:
                adj[rows[k]["bi"]] |= 1 << rows[k]["bj"]

        # reverse topological order: an edge a -> b has x_a < x_b on this axis
        via = [0] * n          # reachable from a by a path of length >= 2
        thru = [0] * n         # reachable from a at all, usable intermediates
        for a in sorted(range(n), key=lambda b: P[b][axis], reverse=True):
            v, m_bits = 0, adj[a]
            while m_bits:
                low = m_bits & -m_bits
                m = low.bit_length() - 1
                m_bits ^= low
                if usable[m]:
                    v |= thru[m]
            via[a], thru[a] = v, adj[a] | v

        for k in idxs:
            if via[rows[k]["bi"]] >> rows[k]["bj"] & 1:
                keep[k] = False

    for k, r in enumerate(rows):
        if not r["terms"] and r["rhs"] >= 0.0:
            keep[k] = False        # `0 <= nonneg`, true without the row
    return keep


def build_and_solve(ci, P, freeze_units, rho=0.06, sep_trim=False, prune_B=None,
                    force_keep=frozenset(), area_R=None, area_g=1.05,
                    area_tol=None, area_price=0.0):
    """prune_B (L112): displacement bound used to drop HPWL terms that provably
    cannot change sign.  None = off = the shipped formulation, bit-for-bit.

    area_R (L122, ported L147): replaces the two-sided band on the LINEARISED
    area with tangent cuts of the convex side plus a price on the non-convex
    one.  None = off = the shipped band, bit-for-bit.  The band is asymmetric in
    practice -- the lower row binds on 259 of 338 reshapeable units and the
    upper on 9 -- and the upper row is itself the barrier: an exact-area
    widening by r has true area p but LINEARISED area p*(r + 1/r - 1), which at
    r=1.5 reads +16.7% against a +/-0.8% band.  area_tol overrides _LP_AREA_TOL
    for the tangent arm only, so the shipped band arm stays the control.

    Each (edge, axis) contributes `t >= |dC + delta|` as one aux column plus two
    rows -- 80.2% of all rows and ~98% of all columns (L112 S1: 15,572 cols /
    38,063 rows on case 99).  But |dC + delta| is only nonlinear where the sign
    can flip.  With |d_u| <= prune_B and |dsize| <= rho*dim, delta is bounded, so
    any edge with |dC| > max|delta| has a FIXED sign: the absolute value is
    linear there and its column+rows are exactly redundant -- fold the linear
    part into the objective instead.  Measured prunable share: 71.5-84.4%.

    Exactness is NOT assumed from "displacements look small": the caller must
    enforce |d_u| <= prune_B (bounds below) and then verify (a) no d_u sits on
    that bound and (b) no dropped term actually flipped.  build returns
    `prune_const` (the folded constant) and `prune_dropped/kept` for that check.
    """
    c = l3.CASES[ci]
    n, cn = c["n"], c["cn"]
    units, unit_of, group_units, group_comp0 = decompose(ci, P)
    U = len(units)
    resh = reshapeable(ci, units)
    if area_R is not None:
        # rho stops being a trust region under area_R and becomes only what its
        # two remaining users need -- an upper bound on |dsize| / dim for the
        # separation reduction mask and the HPWL prune slack.
        rho = max(rho, area_R - 1.0)

    XMIN, XMAX, YMIN, YMAX = 2 * U, 2 * U + 1, 2 * U + 2, 2 * U + 3
    nv = 2 * U + 4
    sv = {}
    for uid in sorted(resh):
        sv[uid] = (nv, nv + 1)
        nv += 2

    obj = [0.0] * nv
    rub, cub, vub = [], [], []
    req, ceq, veq, beq = [], [], [], []
    rows_by_origin = Counter()

    bub = []

    def add_ub(terms, rhs, origin):
        r = len(bub)
        bub.append(rhs)
        for col, coef in terms:
            rub.append(r), cub.append(col), vub.append(coef)
        rows_by_origin[origin] += 1

    def add_eq(terms, rhs, origin):
        r = len(beq)
        beq.append(rhs)
        for col, coef in terms:
            req.append(r), ceq.append(col), veq.append(coef)
        rows_by_origin[origin] += 1

    def new_aux(w):
        nonlocal nv
        obj.append(w)
        nv += 1
        return nv - 1

    def dsize(u, axis):
        if u is None or u not in sv:
            return None
        return sv[u][axis]

    prune_const = 0.0
    prune_stat = [0, 0]                       # [dropped, kept]
    dropped = []                              # (term_id, lin, dC, wsc)
    term_id = [0]

    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        nonlocal prune_const
        tid = term_id[0]
        term_id[0] += 1
        lin = []
        slack = 0.0
        for u, s in ((ui, 1.0), (uj, -1.0)):
            if u is None:
                continue
            lin.append((off + u, s))
            slack += prune_B or 0.0
            k = dsize(u, axis)
            if k is not None:
                lin.append((k, 0.5 * s))
                slack += 0.5 * rho * P[resh[u]][2 + axis]
        if prune_B is not None and abs(dC) > slack and tid not in force_keep:
            # Assume sign(dC + delta) == sign(dC) and fold the term in linearly.
            # |z| >= s*z for either s, so the pruned objective is a LOWER BOUND
            # on the true one everywhere.  Hence if the pruned optimum happens to
            # satisfy every assumed sign, it is optimal for the true problem too
            # -- which is why prune_B needs no bound clamp and is only a
            # HEURISTIC for which terms to try dropping.  solve_pruned() below
            # does that check and regenerates any term whose sign came out wrong.
            sgn = 1.0 if dC > 0.0 else -1.0
            for col, coef in lin:
                obj[col] += wsc * sgn * coef
            prune_const += wsc * sgn * dC
            prune_stat[0] += 1
            dropped.append((tid, tuple(lin), dC, wsc))
            return
        prune_stat[1] += 1
        t = new_aux(wsc)
        t1 = [(t, -1.0)] + lin
        t2 = [(t, -1.0)] + [(col, -coef) for col, coef in lin]
        add_ub(t1, -dC, "hpwl")
        add_ub(t2, dC, "hpwl")

    h_base = max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    hw_scale = 0.5 / h_base
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    const_h = 0.0
    obj0 = 0.0

    b2l_items, p2l_items = _aggregate_pairwise_edges(c, unit_of)
    for i, j, w in b2l_items:
        ui, uj = unit_of[i], unit_of[j]
        dCx, dCy = cx[i] - cx[j], cy[i] - cy[j]
        if w <= 0.0 or ui == uj:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, uj, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, uj, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))

    for p, i, w in p2l_items:
        ui = unit_of[i]
        px, py = c["pin"][p]
        dCx, dCy = cx[i] - px, cy[i] - py
        if w <= 0.0 or ui is None:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, None, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, None, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))

    sep_rows = []
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            ui, uj = unit_of[i], unit_of[j]
            if ui == uj:
                continue
            xj, yj, wj, hj = P[j]
            cands = (
                (xj - (xi + wi), ui, uj, 0, 0, i, j),
                (xi - (xj + wj), uj, ui, 0, 0, j, i),
                (yj - (yi + hi), ui, uj, U, 1, i, j),
                (yi - (yj + hj), uj, ui, U, 1, j, i),
            )
            # key is t[0] only, so the extra block ids cannot change the pick
            gap, ul, ur, off, axis, bl, br = max(cands, key=lambda t: t[0])
            terms = []
            if ul is not None:
                terms.append((off + ul, 1.0))
                k = dsize(ul, axis)
                if k is not None:
                    terms.append((k, 1.0))
            if ur is not None:
                terms.append((off + ur, -1.0))
            sep_rows.append({"axis": axis, "bi": bl, "bj": br,
                             "terms": terms, "rhs": gap})

    keep_mask = (_sep_reduction_mask(sep_rows, n, P, unit_of, sv, resh, rho)
                 if sep_rows else [])
    sep_kept = sum(1 for x in keep_mask if x)
    for row, kf in zip(sep_rows, keep_mask if sep_trim else [True] * len(keep_mask)):
        if kf:
            add_ub(row["terms"], row["rhs"], "separation")

    xmin0 = min(P[i][0] for i in range(n))
    xmax0 = max(P[i][0] + P[i][2] for i in range(n))
    ymin0 = min(P[i][1] for i in range(n))
    ymax0 = max(P[i][1] + P[i][3] for i in range(n))
    W0, H0 = xmax0 - xmin0, ymax0 - ymin0

    def touch_ok(i, code):
        x, y, w, h = P[i]
        return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                and (not code & 8 or abs(y - ymin0) < EPS_BND))

    bnd = [(i, cn[i][4]) for i in range(n) if cn[i][4] != 0]
    sat = [(i, code) for i, code in bnd if touch_ok(i, code)]
    bnd_skip = len(bnd) - len(sat)
    sides = ((1, XMIN, 0, xmin0, min(range(n), key=lambda i: P[i][0]), False, 0),
             (2, XMAX, 0, xmax0, max(range(n), key=lambda i: P[i][0] + P[i][2]), True, 0),
             (4, YMAX, U, ymax0, max(range(n), key=lambda i: P[i][1] + P[i][3]), True, 1),
             (8, YMIN, U, ymin0, min(range(n), key=lambda i: P[i][1]), False, 1))
    for bit, bv, off, ext0, mdef, far, axis in sides:
        tied = {unit_of[i] for i, code in sat if code & bit}
        if not tied:
            continue
        for u in tied:
            if u is None:
                add_eq([(bv, -1.0)], -ext0, "boundary_eq")
            else:
                t = [(off + u, 1.0), (bv, -1.0)]
                k = dsize(u, axis) if far else None
                if k is not None:
                    t.append((k, 1.0))
                add_eq(t, -ext0, "boundary_eq")
        um = unit_of[mdef]
        if um not in tied:
            if um is None:
                add_eq([(bv, -1.0)], -ext0, "boundary_eq")
            else:
                t = [(off + um, 1.0), (bv, -1.0)]
                k = dsize(um, axis) if far else None
                if k is not None:
                    t.append((k, 1.0))
                add_eq(t, -ext0, "boundary_eq")

    for i in range(n):
        ui = unit_of[i]
        if ui is None:
            continue
        x, y, w, h = P[i]
        add_ub([(XMIN, 1.0), (ui, -1.0)], x, "envelope")
        t = [(ui, 1.0), (XMAX, -1.0)]
        k = dsize(ui, 0)
        if k is not None:
            t.append((k, 1.0))
        add_ub(t, -(x + w), "envelope")
        add_ub([(YMIN, 1.0), (U + ui, -1.0)], y, "envelope")
        t = [(U + ui, 1.0), (YMAX, -1.0)]
        k = dsize(ui, 1)
        if k is not None:
            t.append((k, 1.0))
        add_ub(t, -(y + h), "envelope")
    add_ub([(XMAX, 1.0), (XMIN, -1.0)], W0, "bbox")
    add_ub([(YMAX, 1.0), (YMIN, -1.0)], H0, "bbox")

    at = c["at"]
    if area_R is not None:
        # L122: the area band replaced by TANGENT CUTS of w*h >= A(1-TOL).
        # The lower row bounds a CONVEX region (h >= A'/w), so tangents
        # represent it EXACTLY -- no linearisation error and no trust region.
        # The upper row is the non-convex side; it is dropped and adjudicated
        # afterwards by hard_ok, which is the same solve-then-verify contract
        # solve_pruned already uses.
        #
        # Tangent at wk:  h >= 2A'/wk - (A'/wk^2)*w.  Points are geometric with
        # ratio area_g across [w0/R, w0*R]; consecutive tangents cross at
        # wk*sqrt(g), where the envelope sits (sqrt(g)-1)^2 below the curve. At
        # g=1.05 that is 0.061%, so with A' = A*(1-tol) the true area cannot
        # fall below A*(1-0.008)*(1-0.00061) = 0.9914*A -- inside the official
        # 1% hard limit. area_g is therefore the ROW-COUNT knob: steps scales as
        # 1/ln(g), and g=1.10 with tol=0.006 cuts the rows 44% while TIGHTENING
        # the guarantee to 0.99163.
        _tol = _LP_AREA_TOL if area_tol is None else area_tol
        steps = max(1, int(math.ceil(2.0 * math.log(area_R) / math.log(area_g))))
        for uid, i in resh.items():
            kw, kh = sv[uid]
            w, h = P[i][2], P[i][3]
            A = float(at[i]) if float(at[i]) > 0 else w * h
            Ap = A * (1.0 - _tol)
            for s in range(steps + 1):
                wk = (w / area_R) * (area_R * area_R) ** (s / steps)
                sl = Ap / (wk * wk)
                # h0 + dh >= 2A'/wk - (A'/wk^2)(w0 + dw)  ->  -sl*dw - dh <= rhs
                add_ub([(kw, -sl), (kh, -1.0)],
                       -(2.0 * Ap / wk - sl * w - h), "area_tangent")
    else:
        for uid, i in resh.items():
            kw, kh = sv[uid]
            w, h = P[i][2], P[i][3]
            p = w * h
            A = float(at[i]) if float(at[i]) > 0 else p
            slack = rho * rho * p
            add_ub([(kw, -h), (kh, -w)], -(A * (1.0 - _LP_AREA_TOL) - p + slack), "area_band")
            add_ub([(kw, h), (kh, w)], A * (1.0 + _LP_AREA_TOL) - p - slack, "area_band")

    a_base = max(float(c["base"].get("area_baseline", W0 * H0)), 1e-6)
    if W0 * H0 > a_base:
        bA = 0.5 / a_base
        obj[XMIN] -= bA * H0
        obj[XMAX] += bA * H0
        obj[YMIN] -= bA * W0
        obj[YMAX] += bA * W0

    # Deliberately NOT gated on area_R: the price must be applicable to the
    # shipped band too, or there is no control arm separating "wider aspect
    # range" from "shrink every block to its minimum legal area".
    if area_price:
        # The dropped upper bound has exactly one failure mode: a block with no
        # pressure on it runs to the far corner of the shape box, landing at
        # area A*R^2 (measured 44% / 125% / 300% at R=1.2/1.5/2, i.e. R^2-1 to
        # the digit). A tiny price on the block's own area removes the incentive
        # without competing with any real term: {w*h >= A'} is convex, so a
        # positive linear cost pushes such a block ONTO its boundary, where the
        # true area is A' exactly.
        pw = area_price * 0.5 / a_base
        for uid, i in resh.items():
            kw, kh = sv[uid]
            obj[kw] += pw * P[i][3]
            obj[kh] += pw * P[i][2]

    for u in freeze_units:
        add_eq([(u, 1.0)], 0.0, "boundary_eq")
        add_eq([(U + u, 1.0)], 0.0, "boundary_eq")
        if u in sv:
            add_eq([(sv[u][0], 1.0)], 0.0, "boundary_eq")
            add_eq([(sv[u][1], 1.0)], 0.0, "boundary_eq")

    D = W0 + H0 + 1.0
    fro = [i for i in range(n) if unit_of[i] is None]
    # L112: NO clamp. An earlier cut restricted |d_u| <= prune_B to make the
    # pruning bound hold a priori; the lower-bound argument in add_hpwl_rows
    # makes that unnecessary, and clamping would turn the LP into a
    # restriction whose optimum can differ from the real one.
    bounds = [(-D, D)] * (2 * U)
    bounds.append((xmin0 - D, min((P[i][0] for i in fro), default=xmin0 + D)))
    bounds.append((max((P[i][0] + P[i][2] for i in fro), default=xmax0 - D), xmax0 + D))
    bounds.append((ymin0 - D, min((P[i][1] for i in fro), default=ymin0 + D)))
    bounds.append((max((P[i][1] + P[i][3] for i in fro), default=ymax0 - D), ymax0 + D))
    for uid in sorted(sv):
        i = resh[uid]
        if area_R is not None:
            # a box on the SHAPE, not a trust region on the linearisation
            bounds.append((P[i][2] / area_R - P[i][2], P[i][2] * area_R - P[i][2]))
            bounds.append((P[i][3] / area_R - P[i][3], P[i][3] * area_R - P[i][3]))
            continue
        bounds.append((-rho * P[i][2], rho * P[i][2]))
        bounds.append((-rho * P[i][3], rho * P[i][3]))
    bounds += [(0.0, None)] * (nv - len(bounds))

    t_build0 = time.perf_counter()
    A_ub = sparse.csr_matrix((vub, (rub, cub)), shape=(len(bub), nv))
    A_eq = (sparse.csr_matrix((veq, (req, ceq)), shape=(len(beq), nv))
            if beq else None)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    res = linprog(np.asarray(obj), A_ub=A_ub, b_ub=np.asarray(bub),
                  A_eq=A_eq, b_eq=np.asarray(beq) if beq else None,
                  bounds=bounds, method="highs")
    t_solve = time.perf_counter() - t_solve0

    return dict(
        res=res,
        prune_const=prune_const,
        prune_dropped=prune_stat[0],
        prune_kept=prune_stat[1],
        prune_dropped_terms=dropped,
        units=units,
        unit_of=unit_of,
        U=U,
        sv=sv,
        group_units=group_units,
        group_comp0=group_comp0,
        const_h=const_h,
        obj0=obj0,
        rows_ub=len(bub),
        rows_eq=len(beq),
        nnz=len(vub) + len(veq),
        rows_by_origin=dict(rows_by_origin),
        sep_rows_total=len(sep_rows),
        sep_rows_kept=sep_kept,
        timing=dict(t_build=t_build, t_solve=t_solve),
    )


def apply_all(P, B, x):
    U = B["U"]
    out = list(P)
    for uid, mem in enumerate(B["units"]):
        dx, dy = x[uid], x[U + uid]
        dw = dh = 0.0
        if uid in B["sv"]:
            dw, dh = x[B["sv"][uid][0]], x[B["sv"][uid][1]]
        for i in mem:
            px, py, pw, ph = P[i]
            out[i] = (px + dx, py + dy, pw + dw, ph + dh)
    return out


PRUNE_B = None      # L112: module-level so dep_case/mode_ab measure end-to-end


def lp_pass(ci, P, rho, sep_trim=False, **kw):
    c = l3.CASES[ci]
    freeze = set()
    for attempt in range(3):
        t0 = time.perf_counter()
        # HPWL pruning and the separation transitive reduction used to be
        # mutually exclusive here purely because solve_pruned did not forward
        # sep_trim. They are independent reductions -- after pruning,
        # separation is the MAJORITY of the remaining rows (56-73% on the
        # heavy cases), so the combination is the interesting one.
        if PRUNE_B is not None:
            B, _rounds = solve_pruned(ci, P, freeze, rho=rho, prune_B=PRUNE_B,
                                      sep_trim=sep_trim, **kw)
        else:
            B = build_and_solve(ci, P, freeze, rho=rho, sep_trim=sep_trim, **kw)
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}", t=time.perf_counter() - t0,
                              t_build=B["timing"]["t_build"], t_solve=B["timing"]["t_solve"],
                              lp_obj=None, attempts=attempt + 1), None
        newP = apply_all(P, B, B["res"].x)
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(l3.comp_split(newP, [i for i in range(c["n"]) if c["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", t=time.perf_counter() - t0,
                              t_build=B["timing"]["t_build"], t_solve=B["timing"]["t_solve"],
                              lp_obj=float(B["res"].fun) + B.get("prune_const", 0.0),
                              attempts=attempt + 1), B
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break", t=0.0, t_build=0.0, t_solve=0.0, lp_obj=None, attempts=3), None


def solve_pruned(ci, P, freeze_units, rho=0.06, prune_B=None, max_rounds=3,
                 sep_trim=False, **kw):
    """build_and_solve with HPWL pruning made EXACT by verification.

    Dropping `t >= |z|` in favour of `sgn(dC) * z` can only LOWER the objective
    (|z| >= s*z for either s), so the pruned program is a lower bound on the real
    one.  If its optimum satisfies every assumed sign then it attains the real
    objective there, and being <= the real minimum it must BE the real minimum.
    So: solve, check the dropped terms' signs, and re-solve with the offenders
    forced back in.  prune_B is therefore only a guess at which terms are safe --
    a wrong guess costs a round, never correctness.  Falls back to the unpruned
    build if it has not converged in `max_rounds`.
    """
    if prune_B is None:
        return build_and_solve(ci, P, freeze_units, rho=rho,
                               sep_trim=sep_trim, **kw), 0
    keep = set()
    for rnd in range(max_rounds):
        d = build_and_solve(ci, P, freeze_units, rho=rho, prune_B=prune_B,
                            force_keep=keep, sep_trim=sep_trim, **kw)
        res = d["res"]
        if res.status != 0:
            return build_and_solve(ci, P, freeze_units, rho=rho,
                                   sep_trim=sep_trim, **kw), rnd + 1
        x = res.x
        bad = set()
        for tid, lin, dC, _w in d["prune_dropped_terms"]:
            delta = 0.0
            for col, coef in lin:
                delta += coef * x[col]
            # assumed sgn(dC); violated iff the true value took the other branch
            if (dC + delta) * (1.0 if dC > 0.0 else -1.0) < -1e-12:
                bad.add(tid)
        if not bad:
            d["prune_rounds"] = rnd + 1
            return d, rnd + 1
        keep |= bad
    return build_and_solve(ci, P, freeze_units, rho=rho,
                           sep_trim=sep_trim, **kw), max_rounds


_HARD_MASKS = {}


def _hard_masks(ci):
    """Per-case boolean masks for hard_ok, built once."""
    m = _HARD_MASKS.get(ci)
    if m is None:
        c = l3.CASES[ci]
        cn, at, n = c["cn"], c["at"], c["n"]
        fixed = np.array([cn[i][0] != 0 or cn[i][1] != 0 for i in range(n)])
        pre = np.array([cn[i][1] != 0 for i in range(n)])
        area = np.array([float(at[i]) for i in range(n)])
        m = _HARD_MASKS[ci] = (fixed, pre, area, ~fixed & (area > 0))
    return m


def hard_ok(P0, P, ci):
    """Vectorised; same accept/reject as the original nested loops.

    The old version was a pure-Python O(n^2) pair scan -- part of the ~19% of
    tLP that L100's honest-scope section never split out (L112 S1 measured it).
    Early-exit on the first violation is dropped, which only matters for the
    rejected minority; the returned verdict is identical because all the
    comparisons are the same strict float tests over the same pairs.
    """
    fixed, pre, area, soft = _hard_masks(ci)
    A = np.asarray(P, dtype=float)
    A0 = np.asarray(P0, dtype=float)
    if np.any(A[:, 2] <= 0) or np.any(A[:, 3] <= 0):
        return False
    if np.any(fixed & ((A[:, 2] != A0[:, 2]) | (A[:, 3] != A0[:, 3]))):
        return False
    if np.any(pre & ((A[:, 0] != A0[:, 0]) | (A[:, 1] != A0[:, 1]))):
        return False
    if np.any(soft & (np.abs(A[:, 2] * A[:, 3] - area) > 0.01 * area)):
        return False
    x2, y2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    ox = np.minimum(x2[:, None], x2[None, :]) - np.maximum(A[:, 0][:, None], A[:, 0][None, :])
    oy = np.minimum(y2[:, None], y2[None, :]) - np.maximum(A[:, 1][:, None], A[:, 1][None, :])
    bad = (ox > EPS_OVL) & (oy > EPS_OVL)
    np.fill_diagonal(bad, False)
    return not bool(bad.any())


def _lp_build_case(block_count, area_targets, b2b_connectivity,
                   p2b_connectivity, pins_pos, constraints, base):
    """The dict shape the extracted LP reads out of l3.CASES.

    Mirrors l99_oos_shape.build_case, which is how the out-of-sample corpus was
    fed to this same code -- so the shipped assembly is the one that was
    measured, not a re-derivation."""
    n = int(block_count)
    b2l = [(int(e[0]), int(e[1]), float(e[2]))
           for e in b2b_connectivity.tolist() if int(e[0]) != -1]
    p2l = [(int(e[0]), int(e[1]), float(e[2]))
           for e in p2b_connectivity.tolist() if int(e[0]) != -1]
    return dict(
        idx="live", n=n, base=base, at=area_targets, b2b=b2b_connectivity,
        p2b=p2b_connectivity, pins=pins_pos, cons=constraints,
        b2l=b2l, p2l=p2l,
        pin=[(float(p[0]), float(p[1])) for p in pins_pos.tolist()],
        cn=[[int(v) for v in constraints[i].tolist()] for i in range(n)],
    )


_LP_UTIL = 0.968      # L95 structural floor; the label achieves 96.6%


def _shape_lp(pos, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, margs):
    """Post-process the selected layout. Returns `pos` unchanged on any doubt.

    The keep rule is the deployable one measured offline (l100.dep_case): accept
    only if the shapely proxy strictly improves on hpwl or area, does not worsen
    hpwl/area/vrel, and hard_ok passes (positive dims, frozen blocks unmoved,
    preplaced unmoved, 1% area band, no overlap). Decision path touches no
    official evaluator -- same as the shipped selection."""
    key = "live"
    passes = [0]
    try:
        pb = os.environ.get("ICCAD_SHAPE_LP_B", "8")
        prune = None if pb in ("", "0") else float(pb)
    except ValueError:
        prune = 8.0
    # L147 (port of L122): tangent cuts on the area's convex side.
    #
    # SHIPPED BY CODE DEFAULT since L158. It has to be: `make_submission.py
    # verify`, `l113_ship_gate` and the grader all run the official command
    # with ICCAD_* STRIPPED (l113_ship_gate.py:140), so a mechanism that can
    # only be switched on by an environment variable is inert in the package
    # no matter how green it measures. L137 ships the same way, through
    # _l137_env(); the tangent cannot use that path because it is read here in
    # Python rather than by the C++ subprocess, so the default lives in the
    # getenv fallbacks instead. The values are _L147_*, verbatim the ones every
    # L147/L154/L157 arm was measured with.
    #
    # ICCAD_SHAPE_LP_L147=0 is the kill switch and restores the pre-L147
    # shipped band bit-for-bit: it puts every default back to what it was, so
    # `_r` and `_p` come back empty and lpkw ends up {}. A bare
    # ICCAD_SHAPE_LP_R=0 is NOT that switch -- it drops the tangent rows but
    # would leave area_price defaulted, which is a third configuration nobody
    # has measured.
    lpkw = {}
    _on = os.environ.get("ICCAD_SHAPE_LP_L147", "") != "0"
    try:
        _r = os.environ.get("ICCAD_SHAPE_LP_R", _L147_R if _on else "")
        if _r not in ("", "0"):
            lpkw["area_R"] = float(_r)
            lpkw["area_g"] = float(os.environ.get(
                "ICCAD_SHAPE_LP_G", _L147_G if _on else "1.05"))
            _t = os.environ.get("ICCAD_SHAPE_LP_TOL", _L147_TOL if _on else "")
            if _t:
                lpkw["area_tol"] = float(_t)
        _p = os.environ.get("ICCAD_SHAPE_LP_PRICE", _L147_PRICE if _on else "")
        if _p:
            lpkw["area_price"] = float(_p)
        # L150: band-dependent ROW COUNT. The tangent arm's RF cost is a tail --
        # p50 +0.047s but max +1.092s -- and the tail is the big-n cases, which
        # are also the ones with the least runtime slack. rows/unit is
        # ceil(2*ln(R)/ln(g)) + 1, so either knob shrinks it on that band alone.
        # ⚠️ g is bounded by the area guarantee: the tangent envelope sits
        # (sqrt(g)-1)^2 under the curve, and (1-tol)*(1-(sqrt(g)-1)^2) >= 0.99
        # must hold, so g=1.15 needs tol <= 0.0046 and g=1.20 is unusable.
        if lpkw.get("area_R") is not None:
            _bign = int(os.environ.get("ICCAD_SHAPE_LP_BIG_N", "110"))
            if int(block_count) > _bign:
                _rb = os.environ.get("ICCAD_SHAPE_LP_R_BIG", "")
                _gb = os.environ.get("ICCAD_SHAPE_LP_G_BIG", "")
                _tb = os.environ.get("ICCAD_SHAPE_LP_TOL_BIG", "")
                if _rb:
                    lpkw["area_R"] = float(_rb)
                if _gb:
                    lpkw["area_g"] = float(_gb)
                if _tb:
                    lpkw["area_tol"] = float(_tb)
    except ValueError:
        lpkw = {}

    # L157: derive the depth AFTER lpkw, because the gate is coupled to
    # the tangent arm being in play -- see _shape_lp_depth.
    iters, gated = _shape_lp_depth(bool(lpkw))
    _pass_dt = []
    global PRUNE_B
    P0 = [tuple(float(v) for v in p) for p in pos]
    P = P0
    prev = _proxy_metrics(P0, *margs)
    # 🚨 BASELINE-FREE SCALING. Offline, c["base"] carried the evaluator's
    # hpwl_baseline / area_baseline -- and _extract_baseline reads those off the
    # GROUND TRUTH label, so the optimizer cannot see them at solve time. They
    # only set the relative weight of the HPWL and area terms (0.5/base each,
    # mirroring the official cost), so the deployable substitute is the selected
    # layout's OWN hpwl and bbox area: the LP then minimises relative
    # improvement in each term under the official 0.5/0.5 split.
    # ⚠️ This makes it a DIFFERENT objective from the one +2.3559% in-set /
    # +2.4671% OOS was measured on. Those numbers do NOT carry over; the
    # deployable gain has to be measured again through the official eval.
    # area_baseline from the STRUCTURAL FLOOR, not from our own bbox. Our bbox
    # runs ~15% larger than the label's (utilisation 82.2% vs 96.6%, L95), so
    # using it under-weights the area term by that much -- and the area term is
    # where the shape LP's whole upside lives. sum(A)/_LP_UTIL is label-free and
    # per-case. Measured against the label baselines the LP cannot see:
    # own -> 92.6 / 89.4 / 85.0 % of the oracle gain at k = 1 / 4 / 12;
    # this -> 100.0 / 100.7 / 100.3 %. Flat for _LP_UTIL anywhere in
    # [0.85, 1.05] (all within 6e-6 of each other), so it is not a fitted knob.
    _sumA = sum(max(0.0, float(area_targets[i])) for i in range(int(block_count)))
    base = {"hpwl_baseline": max(float(prev["hpwl"]), 1e-6),
            "area_baseline": max(_sumA / _LP_UTIL, 1e-6)}
    l3.CASES[key] = _lp_build_case(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints, base)
    saved, PRUNE_B = PRUNE_B, prune

    def _chain(kw):
        """One accept-guarded LP chain from P0. Returns the layout it kept.

        Factored out of the loop below unchanged so that the L154 band-catch
        can re-run the SAME guard with different rows -- a second tier must be
        adjudicated by the same rule as the first or it is a different keep
        rule wearing the same name.
        """
        P_, prev_ = P0, prev
        dt_pass = 0.0
        for _it in range(max(1, iters)):
            # L157: the first pass is never gated -- it is what ships.
            # From the second on, spend only if this case can absorb it
            # and stay on the RF floor. dt_pass is the previous pass
            # MEASURED, not a fitted cost.
            if _it and gated and not _depth_affordable(block_count, dt_pass):
                break
            # sep_trim=True: the separation transitive reduction. Exact
            # (a removed row stays implied by the ones that survive) and
            # independent of the HPWL pruning -- they used to be mutually
            # exclusive only because solve_pruned did not forward the flag.
            # After pruning, separation is 56-73% of the remaining rows on
            # the heavy cases. Measured over 100 cases, min of 3:
            # weighted tLP 0.2670s -> 0.1947s (1.37x) with the quality
            # identical to every digit (1.236783247, kept 100/100).
            _t_pass = time.perf_counter()
            newP, tele, _B = lp_pass(key, P_, 0.06, sep_trim=True, **kw)
            dt_pass = time.perf_counter() - _t_pass
            passes[0] += 1
            # L159: per-PASS timing, inside one run. Cross-run
            # differencing cannot measure this: the null control (the 25
            # n the gate excludes, where two arms do identical work) drifts
            # -5.03s over 25 cases = 0.20s/case, against a signal of
            # 0.08s/case. Timing both passes in the SAME process removes
            # the drift entirely instead of trying to average it away.
            if _LP_TIMING:
                _pass_dt.append(dt_pass)
            if tele["lp_obj"] is None:
                break
            m = _proxy_metrics(newP, *margs)
            better = (m["hpwl"] < prev_["hpwl"] * (1 - 1e-12)
                      or m["area"] < prev_["area"] * (1 - 1e-12))
            worse = (m["hpwl"] > prev_["hpwl"] * (1 + 1e-12)
                     or m["area"] > prev_["area"] * (1 + 1e-12)
                     or m["vrel"] > prev_["vrel"] + 1e-12)
            if not (better and not worse and hard_ok(P0, newP, key)):
                break
            P_, prev_ = newP, m
        return P_

    # L154 BAND-CATCH (off unless ICCAD_SHAPE_LP_CATCH=1). A tangent-arm case
    # that hard_ok refuses currently falls all the way back to the pre-LP
    # layout -- so it loses the whole SHIPPED LP gain, not just the tangent
    # increment. L153 measured that on the Linux verify: case 96 (n=117) is
    # rejected there and kept on Windows, and that one case is 107% of the
    # Windows/Linux spread (the shipped band keeps it at 1.186644; rejection
    # returns 1.215357). The shipped band's own layout is a legal second tier:
    # it is what ships today, it is adjudicated by the same guard, and a case
    # that reaches tier 2 lands on the shipped-band result bit-for-bit.
    # The mean is small; the point is variance -- it bounds the downside of
    # every rejection at "no worse than what is already deployed".
    _catch = os.environ.get("ICCAD_SHAPE_LP_CATCH", "") == "1"
    tier = 0
    try:
        P = _chain(lpkw)
        if P is not P0:
            tier = 1
        elif _catch and lpkw:
            # only worth a second solve when the first chain ran NON-shipped
            # rows; with lpkw empty the retry would be the identical program.
            P = _chain({})
            tier = 2 if P is not P0 else 0
    finally:
        PRUNE_B = saved
        l3.CASES.pop(key, None)
        # _HARD_MASKS is keyed by case id and every live call uses the
        # SAME key, so a stale entry would hand the next case masks sized
        # for the previous one (numpy broadcast error, 99/100 cases).
        _HARD_MASKS.pop(key, None)
    global _LAST_PASS_DT
    _LAST_PASS_DT = list(_pass_dt)
    kept = P is not P0
    # L147 diagnostic, off by default: one line per case, appended. The tangent
    # arm drops the upper area bound and leaves hard_ok to adjudicate, and a
    # REJECTED case loses the whole shipped LP gain, not just the increment --
    # so kept-rate is the gate, and it has to be observable.
    # L154 added the third field, the TIER that kept it: 0 rejected (pre-LP),
    # 1 the requested rows, 2 the band-catch. Without it a band-catch run is
    # indistinguishable from a run where the tangent rows simply got luckier.
    _sp = os.environ.get("ICCAD_SHAPE_LP_STATS", "")
    if _sp:
        try:
            with open(_sp, "a") as fh:
                # L157 4th field: LP passes actually SPENT. Without it a
                # gated run and an ungated one are indistinguishable in
                # the telemetry -- the gate could be a silent no-op and
                # every table above would print the same. l117 indexes
                # this file by position and tolerates the extra column.
                fh.write(f"{int(block_count)} {int(kept)} {int(tier)} "
                         f"{int(passes[0])}\n")
        except Exception:
            pass
    return P if kept else pos


# L157: SELECTIVE LP DEPTH. The RF floor -- max(0.7, R^0.3) on the runtime
# ratio -- is not one global budget. Measured per case against the published
# beta medians the slack runs 0.96x to 3.91x (p50 1.74x), so a mechanism that
# is unaffordable "on average" is free on most cases and ruinous on a few.
# A second LP pass costs p50 0.165s and buys +0.5967% in set, +0.7518% (s1)
# and +0.9959% (s2) out of sample. Spent on EVERY case it prices at NET
# -0.4593%; spent only where the case can absorb it, NET +0.4275%~+0.8289%
# across both OOS samples and both gate forms, against a 0.30% bar.
# Full derivation and the chain of custody: L157_REPORT.md.
# L147 tangent-cut configuration, shipped by code default (see _shape_lp).
# Verbatim the arm measured at in-set +2.5881%, OOS +2.2416% (L147), Linux
# verified at L153, and the base every L154/L157 number sits on top of.
_L147_R, _L147_G, _L147_TOL, _L147_PRICE = "1.5", "1.10", "0.006", "1.0"

# L157 n-set: the block counts whose beta row can absorb a second LP
# pass and stay under 0.3046*M_hat(n). Derived by l157_selective_depth.py
# from the published beta medians; the excluded values inside the span are the
# cases with no slack, not gaps in the corpus.
#
# L160 WIDENED THIS 75 -> 89. The original derivation charged the second pass in
# DEV-BOX seconds against a GRADER-second budget -- a units error that overcharged
# it by f. f is now measured at 2.71 (L160: the beta package M73 re-run from git
# at 7f38893, weighted total 1.295547821428148, bit-identical to the recorded
# beta identity; 141.07s here against the grader's 52.07s, per-case p25 2.33 /
# p50 2.71 / p75 3.20, flat across n so the max-setter premise holds). At f=2.71
# the affordable set is 89, and the old 75 are a strict subset of it.
# Worth, from the committed OOS arms: s1 +0.4882% -> +0.5513%, s2 +0.5673% ->
# +0.6798%, RF unchanged (-0.0661% -> -0.0664%), and 0 of the 14 newly included
# n got worse on either sample.
# ⚠️ This is a bet, not a free lunch: 89 overtakes 75 only above f = 1.47, and
# at f = 1.00 it LOSES 0.21pp. The measured 2.71 and the observed per-case
# minimum 1.79 are both above 1.47, which is the whole case for widening.
_L157_NSET = frozenset((
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 86, 87,
    88, 89, 90, 91, 93, 95, 96, 97, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 119,
))

_M157_A, _M157_B = 0.0196, 1.168   # M_hat(n) = A * n**B, R^2 = 0.907 on beta
_M157_THR = 0.3046                 # 0.7 ** (1/0.3): where max(0.7, R^0.3) lifts
_T_CASE = threading.local()        # per-case clock, stamped by solve()


def _case_clock_start() -> None:
    """Stamp the start of THIS case. Thread-local: the profile pool runs in
    threads but _shape_lp is reached on the same thread that entered solve(),
    so a plain module global would be correct today and would race the day a
    caller drives cases concurrently."""
    _T_CASE.t0 = time.perf_counter()


def _shape_lp_depth(tangent_on=False):
    """(passes allowed, gate the extra ones?) -- L157.

    An explicit ICCAD_SHAPE_LP_ITERS keeps its old meaning and is UNGATED:
    that is how the k=1 and k=2 arms were measured and they have to stay
    reproducible bit-for-bit. ICCAD_SHAPE_LP_DEPTH2=0 forces the shipped k=1
    -- the kill switch, and the control for the bit-equality gate.

    `tangent_on` is the coupling that keeps the DEPLOYED configuration
    equal to the PRICED one. Every L157 number -- in set and on both OOS
    samples -- was measured with L147's tangent rows in play; depth 2 on
    the plain shipped band was never measured at all, and a package with
    no ICCAD_* set would otherwise run exactly that unmeasured arm.
    Gated on lpkw and not on the env var, so it still holds the day the
    tangent is switched on by a code default rather than by the
    environment."""
    v = os.environ.get("ICCAD_SHAPE_LP_ITERS", "")
    if v != "":
        try:
            return max(1, int(v)), False
        except ValueError:
            return 1, False
    if not tangent_on:
        return 1, False
    if os.environ.get("ICCAD_SHAPE_LP_DEPTH2", "") == "0":
        return 1, False
    return 2, True


def _depth_affordable(n, dt_next) -> bool:
    """Can this case absorb dt_next more seconds and stay on the RF floor?

    t_case and dt_next are both OBSERVED -- the case clock, and the pass that
    just ran (a further pass repeats the same build+solve on a slightly
    different P, so the last pass is the right estimate for the next one).
    Only M_hat is estimated; substituting it for the true beta median moves
    the selection from 71 cases to 75 and costs -0.0201% of RF.

    No clock -> False, i.e. shipped k=1. A caller that never went through
    solve() (a probe) must not silently buy the ungated k=2, which prices at
    NET -0.4593%. Every failure direction here falls back to what ships."""
    # DEFAULT: the DETERMINISTIC form. The per-case clock below is a better
    # mechanism on paper -- it reads this box's real time instead of trusting a
    # corpus median -- and it cost this project a standing rule to find out why
    # it cannot ship. CLAUDE.md, twice: "any in-window LP must not keep a HiGHS
    # time_limit (triggering it makes the LP run-to-run nondeterministic)". A
    # wall-clock gate reintroduces exactly that, by a different route. Measured:
    # two runs of identical code and flags decided 5 block counts differently
    # and moved 4 cases, a weighted delta of -0.0011%. The SCORE impact is
    # noise; what it breaks is every bit-equality gate the project verifies
    # with -- make_submission.py verify and l113_ship_gate G4 both compare
    # positions bit-for-bit against an anchor, and both would start failing
    # intermittently. A deterministic n-set keeps them working, and it is the
    # bracket L157_REPORT.md quotes anyway: OOS NET +0.4275% / +0.5066%,
    # against +0.5477% / +0.8289% for the clock form. Buying 0.12-0.32pp with
    # the project's whole verification story is not a trade worth making.
    if os.environ.get("ICCAD_SHAPE_LP_DEPTH_PC", "") != "1":
        return int(n) in _L157_NSET

    # ICCAD_SHAPE_LP_DEPTH_PC=1: the per-case clock form. Measured and
    # documented, NOT shipped. Keep it for the day the grader's own timings
    # are known, or the day invariant gates replace bit-equality ones.
    t0 = getattr(_T_CASE, "t0", 0.0)
    if not t0:
        return False
    m_hat = _M157_A * (max(1.0, float(n)) ** _M157_B)
    # ICCAD_SHAPE_LP_DEPTH_S: machine-speed scale, default 1.0 = OFF.
    # The gate is stated in ABSOLUTE seconds, because R = t/M is measured on
    # whatever box runs the case and M_hat is the grader's median -- correct
    # on the grader, and unfirable anywhere slower. This dev box runs ~9x
    # slower per case (490.7s vs beta's 52.07s over the same 100 cases), so
    # at S=1 the gate is a no-op HERE and the mechanism cannot be exercised
    # end to end. S makes it testable. It is NOT a tuning knob: shipping any
    # value other than 1.0 would be claiming to know the grader's speed.
    try:
        s = float(os.environ.get("ICCAD_SHAPE_LP_DEPTH_S", "1") or "1")
    except ValueError:
        s = 1.0
    return (time.perf_counter() - t0) + dt_next <= _M157_THR * m_hat * s


def _shape_lp_on() -> bool:
    """Cores-gated, on the SAME gate as route A -- deliberately.

    The +2.18% is priced as a weak win only for the PAIR: route A shortens
    the wall enough that the LP's added time lands under the RF floor on
    most cases. LP ALONE is negative from s=1.5 up (-0.75% at 1.5, -2.43%
    at 2), so firing it on a box where route A does not fire would be a
    straight loss. _effective_cores_hi() reports 0 on a detection failure,
    so that direction falls back to shipped behaviour too.
    ICCAD_SHAPE_LP=0 forces it off, =1 forces it on."""
    if not _LP_IMPORTS_OK:
        return False
    v = os.environ.get("ICCAD_SHAPE_LP", "")
    if v != "":
        return v == "1"
    return _effective_cores_hi() >= _ROUTE_A_CORES_MIN


# OFFLINE INSTRUMENT (ported from the teammate's L140, `l113-route-a`). Never
# fires unless ICCAD_LP_TIMING is set; the shipped path pays one os.environ
# lookup per case and is otherwise bit-identical.
#
# 🚨 WHY CPU TIME AND NOT WALL -- their finding, and it invalidates how L147,
# L154 and L157 were all priced. Differencing whole-eval wall on a dev box put
# their OOS 240 at k=2 FASTER than k=1 (1507s vs 1601s) for strictly more work,
# i.e. a NEGATIVE LP cost that looked perfectly reasonable in a table. The same
# noise made their per-pass figure wrong by 2.4x (0.186s against a true
# 0.4446s). 51 portfolio subprocesses run concurrently with everything else on
# the box; process_time() counts only this process's own CPU, and the LP runs
# synchronously in the main process, so it is the right clock.
_LP_TIMING = os.environ.get("ICCAD_LP_TIMING", "") not in ("", "0")
_LAST_PASS_DT = []


def _shape_lp_maybe(pos, *a):
    """Never raises, never returns anything the guard did not accept.

    Knob off, scipy/shapely absent, or ANY exception -> the selected layout is
    returned exactly as it was. Same three-layer doctrine as solve(): the
    post-processing may decline, it may never take the case down with it."""
    if not _shape_lp_on():
        return pos
    if _LP_TIMING:
        _c0, _w0 = time.process_time(), time.perf_counter()
        try:
            _r = _shape_lp(pos, *a)
        except Exception as exc:
            print(f"[constructive] shape LP raised {exc!r}; keeping the selected "
                  f"layout", file=sys.stderr)
            _r = pos
        print(f"[lptime] n={len(pos)} cpu={time.process_time()-_c0:.6f} "
              f"wall={time.perf_counter()-_w0:.6f} "
              f"passes={','.join(f'{d:.6f}' for d in _LAST_PASS_DT)}",
              file=sys.stderr, flush=True)
        return _r
    try:
        return _shape_lp(pos, *a)
    except Exception as exc:
        print(f"[constructive] shape LP raised {exc!r}; keeping the selected "
              f"layout", file=sys.stderr)
        return pos
