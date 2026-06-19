#!/usr/bin/env python3
"""
Constructive-placer PORTFOLIO wrapper.

Drives constructive.exe (C++ port of the teammate's constraint-aware constructive
floorplanner). Runs several deterministic profiles in parallel and selects the
best with a BASELINE-FREE proxy of the contest cost:

    cost  = 0.5*(area/A + hpwl/H) * exp(2*vrel)
    proxy = (area/Â + hpwl/hmin) * exp(2*vrel)     (Â = 1.035*ΣblockArea, hmin =
                                                    min hpwl over profiles)

vrel is exact from (positions, constraints); area/hpwl are emitted by the C++ on
stderr ("METRICS area hpwl vbd vcl vmb nsoft"). Offline the proxy matched the
oracle ceiling almost exactly (1.6060 vs 1.6057) because constructive is
deterministic — no SA timing noise. Profiles vary boundary aspect (the highest-
leverage diversity axis) plus wire/anchor weights via env knobs.

Single base profile ~1.658; portfolio ~1.536 (C++ M9 two-pass wire refinement +
14th frame_fine profile). Set ICCAD_CONSTRUCTIVE_SINGLE=1 to run only the base
profile. ICCAD_CONSTRUCTIVE_BIN overrides the binary path.
"""
import concurrent.futures
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import (
    FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b,
)
from optimizer_claude import _serialize_input, _parse_output, python_sa_solve

try:
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _unary_union
    _SHAPELY = True
except Exception:
    _SHAPELY = False

_BIN = Path(os.environ.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive.exe")))

# Profiles validated by portfolio_ceiling.py. Two diversity axes dominate:
# block boundary-aspect (high LR -> low vBd on violation-heavy cases) and frame
# outline shape (frame_tall wins 13% of weight). Adding profiles is downside-
# protected: the proxy picks per-case, so a never-best profile costs only runtime.
# Portfolio ~1.5362 with this 14-profile set: C++ M9 wire refinement (1.5659->
# 1.5375) + frame_fine (tighter outline scales for area-dominated cases, a further
# marginal -0.08%; the frame-based area lever is near-exhausted -- see dbg_area.py
# and CLAUDE.md). Dropped as useless: wire_xhi, frame_wide, frame_wwire.
#
# M25 audit (profile_audit.py, 2026-06-12, on the M24 jump binary): 18 profiles
# with 0 proxy wins AND leave-one-out dTotal == 0 over all 100 cases are pruned
# below (tagged [M25-pruned]; kept as comments — re-add for hidden-test diversity
# if ever desired). Pruned-pool selection is bit-identical (1.3862). Note the
# per-case wall is dominated by the OS16 profiles' own runtime (max term), so
# pruning cheap knob profiles trims total CPU/contention, not the wall model.
_PROFILES: List[Dict[str, str]] = [
    {},                                                                       # base
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
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},                    # aspect_xhi
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
    {"ICCAD_LR_ASPECT": "3.0"},                                               # lr30 (M32: case 71 ->1.2760)
    {"ICCAD_TB_ASPECT": "0.8"},                                               # tb08 (M32: case 49)
    {"ICCAD_TB_ASPECT": "0.667"},                                             # tb0667 (M32: case 67/52)
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_WIRE_MULT": "2.0"},  # asp_wire
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},                   # aspect_v7
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},                 # aspect_v10  [M25-pruned]
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # asp7_wire
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},   # asp5_anclo
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33"},                           # frame_tall  [M25-pruned] (its combos below carry the axis)
    {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20"},                            # frame_tight
    {"ICCAD_FRAME_SCALES": "1.04,1.07,1.10,1.13,1.16,1.35,1.65,2.10"},        # frame_fine
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10", "ICCAD_WIRE_MULT": "2.0"},  # asp10_wire  [M25-pruned]
    {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},  # tall_asp5
    # {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_WIRE_MULT": "2.0"},   # asp5_wire  [M25-pruned]
    {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_WIRE_MULT": "2.0"},            # tall_wire
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04"},  # asp7_anclo  [M25-pruned]
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_ANCHOR_W": "0.04"},  # asp_anclo
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},  # tall_asp7  [M25-pruned]
    # {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},   # tight_asp5  [M25-pruned]
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "3.0"},  # asp7_wirex3
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},  # tall_asp10  [M25-pruned]
    # [M30-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_ANCHOR_W": "0.04"},  # tall_anclo
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # asp5_all
    # {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10", "ICCAD_WIRE_MULT": "3.0"},  # asp10_wirex3  [M25-pruned]
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # asp7_all  [M25-pruned]
    # [M32-pruned: 0 wins, LOO 0; subsumed by decoupled lr45/lr30] {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp5_wire
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp7_wire  [M25-pruned]
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},   # tall_asp5_anc  [M25-pruned]
    {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04"},  # tall_asp7_anc
    # {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "3.0"},  # asp7_all_x3  [M25-pruned]
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20",  "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "3.0"},  # asp5_all_x3
    # {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp7_all  [M25-pruned]
    {"ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20",  "ICCAD_ANCHOR_W": "0.04", "ICCAD_WIRE_MULT": "2.0"},  # tall_asp5_all
    # {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # tight_asp7_wire  [M25-pruned]
    # [M30r2-pruned: 0 wins, LOO 0] {"ICCAD_FRAME_SCALES": "1.00,1.05,1.10,1.20", "ICCAD_WIRE_MULT": "2.0"},  # tight_wire
    # M13: narrower-than-frame_tall outlines (aspect 0.55-0.28) attack the systematic
    # horizontal dead space (dbg_area: w/wb~1.3-1.5, h/hb~1.0 -> we pack too wide).
    # Wins the highest-weight cases 98 (n=119) and 87 that frame_tall (0.67-0.33)
    # didn't reach. Downside-protected by the proxy.
    {"ICCAD_FRAME_ASPECTS": "0.55,0.45,0.35,0.28"},                                                     # narrow
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
    {"ICCAD_WIRE_BFS": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},                      # bfs_wt_wire
    # [M32-pruned: 0 wins, LOO 0; subsumed by decoupled lr45/lr30] {"ICCAD_WIRE_BFS": "1", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_WIRE_MULT": "2.0"},      # bfs_tall_wire
    # {"ICCAD_WIRE_TIEBREAK": "1", "ICCAD_FRAME_ASPECTS": "0.67,0.5,0.4,0.33", "ICCAD_ANCHOR_W": "0.04"}, # wtb_tall_anc  [M25-pruned] (case 87 superseded since M18)
    # M19: BFS_PIN seeds the BFS attachment with p2b pin weights too (pins are
    # fixed anchors exactly like preplaced blocks). bfs_pin_wt_wire +0.269%
    # oracle-min: re-breaks case 95 (1.2995->1.2767) and takes 94 (1.3656->1.3411)
    # + 64. bfs_tight_wire +0.061%: case 91 (1.3848->1.3712, untouched by PIN)
    # + small-n cases. BFS+anc/PIN+W2/PIN+tall were dominated by these two.
    {"ICCAD_WIRE_BFS": "1", "ICCAD_BFS_PIN": "1", "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"},  # bfs_pin_wt_wire
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
]
_RH = 1.4  # relative weight of the hpwl term in the proxy. The proxy uses hmin
           # (min hpwl over profiles) as a stand-in for the unknown baseline hpwl
           # hbase; since we never beat baseline, hmin > hbase by ~hmin/hbase≈1.3-1.4,
           # so the raw proxy under-weights hpwl vs the true cost (which divides by
           # hbase). _RH≈1.4 compensates -> proxy selection matches the oracle ceiling.
           # Flat basin 1.3-1.6 all hit oracle (1.4349); 1.0 gave 1.4369. (M13 _rh_sweep)


def _ensure_compiled() -> bool:
    src = _DIR / "constructive.cpp"
    if _BIN.exists() and _BIN.stat().st_mtime >= src.stat().st_mtime:
        return True
    for gpp in (r"C:\msys64\ucrt64\bin\g++.exe", "g++"):
        try:
            r = subprocess.run(
                [gpp, "-O3", "-std=c++17", "-o", str(_BIN), str(src)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return True
            print(f"[constructive] compile failed:\n{r.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"[constructive] compile error with {gpp}: {e}", file=sys.stderr)
    return _BIN.exists()


def _proxy_metrics(positions, area_targets, b2b, p2b, pins, constraints, n):
    """Baseline-free (area, hpwl, vrel), computed EXACTLY like the harness so the
    live selector matches the offline-validated proxy. The C++ emits its own vrel
    too, but its union-find grouping (1e-3 tol) disagrees with shapely on ~34% of
    cases; replicating the harness here recovers the oracle-level selection."""
    xmin = min(p[0] for p in positions); ymin = min(p[1] for p in positions)
    xmax = max(p[0] + p[2] for p in positions); ymax = max(p[1] + p[3] for p in positions)
    area = (xmax - xmin) * (ymax - ymin)
    hpwl = calculate_hpwl_b2b(positions, b2b) + calculate_hpwl_p2b(positions, p2b, pins)

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    vb = vg = vm = 0
    nsoft = 0
    if ncols > 4:
        bound = constraints[:n, 4]; clust = constraints[:n, 3]; mib = constraints[:n, 2]
        nsoft = int((bound != 0).sum().item())
        eps = 1e-6
        for i in range(n):
            code = int(bound[i].item())
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
        ngrp = int(clust.max().item()) if clust.numel() else 0
        for g in range(1, ngrp + 1):
            idx = [i for i in range(n) if int(clust[i].item()) == g]
            nsoft += max(0, len(idx) - 1)
            if len(idx) > 1 and _SHAPELY:
                u = _unary_union([_box(positions[i][0], positions[i][1],
                                       positions[i][0] + positions[i][2],
                                       positions[i][1] + positions[i][3]) for i in idx])
                if u.geom_type == "MultiPolygon":
                    vg += len(u.geoms) - 1
        nmib = int(mib.max().item()) if mib.numel() else 0
        for g in range(1, nmib + 1):
            idx = [i for i in range(n) if int(mib[i].item()) == g]
            nsoft += max(0, len(idx) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in idx}
            vm += len(shapes) - 1
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return {"area": area, "hpwl": hpwl, "vrel": vrel}


def _run_profile(env_over: Dict[str, str], inp: str, n: int):
    """Run one profile; return positions or None."""
    env = dict(os.environ)
    env.update(env_over)
    try:
        r = subprocess.run([str(_BIN)], input=inp, capture_output=True,
                           text=True, timeout=120.0, env=env)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return _parse_output(r.stdout, n)
    except Exception:
        return None


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
        if not self._ok:
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        inp = _serialize_input(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions, gnn_hint=None,
        )
        profiles = _PROFILES[:1] if self._single else _PROFILES

        if len(profiles) == 1:
            positions_list = [_run_profile(profiles[0], inp, block_count)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as ex:
                futs = [ex.submit(_run_profile, p, inp, block_count) for p in profiles]
                positions_list = [f.result() for f in futs]

        cands = [pos for pos in positions_list if pos is not None]
        if not cands:
            print("[constructive] all profiles failed; python SA fallback",
                  file=sys.stderr)
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        if len(cands) == 1:
            return cands[0]

        # Baseline-free proxy selection: cost ~ (area/A + hpwl/H)*exp(2*vrel).
        metrics = [_proxy_metrics(pos, area_targets, b2b_connectivity,
                                  p2b_connectivity, pins_pos, constraints, block_count)
                   for pos in cands]
        sumA = sum(max(0.0, float(area_targets[i])) for i in range(block_count))
        A_hat = 1.035 * max(sumA, 1e-9)
        hmin = min(m["hpwl"] for m in metrics) or 1.0
        best_pos, best_proxy = cands[0], float("inf")
        for pos, m in zip(cands, metrics):
            proxy = (m["area"] / A_hat + _RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            if proxy < best_proxy:
                best_proxy, best_pos = proxy, pos
        return best_pos
