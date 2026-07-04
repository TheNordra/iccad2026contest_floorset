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

from iccad2026_evaluate import FloorplanOptimizer
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
]

# M42 (2026-06-26): 2nd-order RuntimeFactor lever — indices into _PROFILES of the
# BUILD-time profiles that are NEVER the selected winner on any case with n>100
# (per-big-n redundancy; the M41 swap argument applied to the build profiles).
# After M41 drops the 6 swap profiles, the big-case (n>=110, 60% weight) wall is
# sum/cores-bound by the ~20 expensive (~8s) FREE/CA/FC profiles; these 21 win NO
# n>100 case, so dropping them there is wall-only (rf_score_model.py: local RF=1.0
# stays 1.3277 BIT-IDENTICALLY — every capped case keeps Qcap/Qbase=1.0000 — while
# the n=120 wall halves 15.6->8.0s, projecting a FURTHER ~-11% real total @ M=11,
# robust over median in [6,20]s and all 20 n>100 cases stay median-INDEPENDENT
# WINs). The kept 13 build profiles are the cluster/anchored/MIB stacks that
# structurally dominate big cases + a few aspect/free winners. REGENERATE after any
# _PROFILES edit: rf_score_model.py M42 section prints the recommended set.
_BIG_REDUNDANT_IDX = frozenset({0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 20,
                                24, 28, 29, 30, 31, 32, 33})

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
_M45_BAND_DROP: Tuple[Tuple[int, int, frozenset], ...] = (
    # mid cases: 9 profiles win nothing in n=61..100 (40 cases, kept 25); several
    # (#2/#13/#17/#18) are n>110 winners — band scoping is what frees them here.
    # sum-bound at any cores -> universal. Projected -1.2..-1.5% real @ M=11.
    (60, 100, frozenset({2, 8, 13, 17, 18, 20, 28, 30, 33})),
)
_M45_LOWCORE_DROP: Tuple[Tuple[int, int, frozenset], ...] = (
    # small band: deep floor at 12c (gain ~0) but sum-bound and scoring at low
    # cores; drops 20/34, kept 14 (strict-equal over its 20 cases).
    (40, 60, frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14,
                        24, 25, 26, 27, 29, 31, 32, 33})),
    # (100,110]: kept 8 — #17/#18 set some walls even at 12c, but diversity floor
    # + high weight keeps this tier-4 (conservative). #19 stays (band winner).
    (100, 110, frozenset({2, 13, 17, 18, 21})),
    # (110,inf): D4 — #19 is the case-90 winner (exact tie with #21, proxy
    # 2.766374981 / cost 1.330092901 both) and is KEPT; only true non-winners go.
    (110, 10**9, frozenset({8, 12, 26, 27})),
)
# tier-4 activation: at 8 detected cores the tier still adds ~-2.2% real @ M=11
# (some walls are sum-bound there) and the strict gate makes it weakly winning at
# ANY cores; 8 also covers 8-vCPU/4-physical cloud boxes where the true gain is
# larger. Above 8 the increment (~-0.9% @12c) is not worth the kept-8 diversity
# squeeze on the high-weight (100,110] band.
_M45_CORES_MAX = 8


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


def _pool_indices(block_count: int) -> List[int]:
    """Kept _PROFILES indices for this case size under the adaptive-pool tiers
    (M41 swap / M42 big-redundant / M45 band + low-core). ICCAD_ADAPTIVE_POOL=0
    returns the full pool."""
    full = list(range(len(_PROFILES)))
    if os.environ.get("ICCAD_ADAPTIVE_POOL", "1") == "0":
        return full
    n_swap = int(os.environ.get("ICCAD_ADAPTIVE_N", "0"))
    n_free = int(os.environ.get("ICCAD_ADAPTIVE_FREE_N", "100"))
    drop_band: frozenset = frozenset()
    if os.environ.get("ICCAD_ADAPTIVE_BAND", "1") != "0":        # M45 tier-3
        for lo, hi, d in _M45_BAND_DROP:
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
        if block_count > n_swap and ("ICCAD_ORDER_SWAP" in p
                                     or "ICCAD_ORDER_MOVE" in p):
            continue
        if block_count > n_free and i in _BIG_REDUNDANT_IDX:
            continue
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
    src = _DIR / "constructive.cpp"
    if not src.exists():
        return _BIN.exists() and _binary_runs()
    if (_BIN.exists() and _BIN.stat().st_mtime >= src.stat().st_mtime
            and _binary_runs()):
        return True
    # M48 compile chain: the first candidate is the exact command used through
    # M47 (identical local behaviour); the others only matter where it is
    # missing (e.g. a Linux grader). -O2 is retried last in case -O3 fails.
    # ICCAD_CXX forces a specific compiler to the front of each round.
    compilers = [r"C:\msys64\ucrt64\bin\g++.exe", "g++", "clang++", "c++"]
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
        try:
            return self._solve_impl(block_count, area_targets, b2b_connectivity,
                                    p2b_connectivity, pins_pos, constraints,
                                    target_positions)
        except Exception as e:
            print(f"[constructive] solve raised {e!r}; python SA fallback",
                  file=sys.stderr)
        try:
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        except Exception as e:
            print(f"[constructive] SA fallback raised {e!r}; row fallback",
                  file=sys.stderr)
            return _row_fallback(block_count, area_targets, constraints,
                                 target_positions)

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
        inp = _serialize_input(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions, gnn_hint=None,
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
        #  M42 (ICCAD_ADAPTIVE_FREE_N, default 100): ALSO drop the 21 _BIG_REDUNDANT_IDX
        #    build-time profiles for block_count>100 — they win NO n>100 case (the swap
        #    argument applied per-big-n), so dropping them is wall-only: RF=1.0 local
        #    stays 1.3277 BIT-IDENTICALLY while the n=120 wall halves 15.6->8.0s,
        #    projecting a FURTHER ~-11% real @ M=11 (1.2904->1.1473), all 20 n>100
        #    cases median-INDEPENDENT WINs, robust over median in [6,20]s.
        # Default ON; ICCAD_ADAPTIVE_POOL=0 restores the full 40-profile pool (local
        # 1.3269). Set ICCAD_ADAPTIVE_FREE_N huge (e.g. 9999) for M41-only behaviour.
        #  M45 (2026-07-02): two more tiers inside _pool_indices() — tier-3 band-
        #    scoped mid-case drops (UNIVERSAL, ICCAD_ADAPTIVE_BAND=0 disables) and
        #    tier-4 low-core drops (only when _effective_cores() <= _M45_CORES_MAX;
        #    ICCAD_ADAPTIVE_CORES forces/disables detection). Both under the strict
        #    selection-preserving gate -> local RF=1.0 score unchanged (1.3277).
        if not self._single:
            profiles = [_PROFILES[i] for i in _pool_indices(block_count)]

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
            return cands[0]

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
            return best_pos
        except Exception as e:
            print(f"[constructive] proxy selection raised {e!r}; keeping first "
                  f"candidate", file=sys.stderr)
            return cands[0]
