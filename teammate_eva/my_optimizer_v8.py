#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet - Legal Baseline v8: Constraint-Aware Greedy Placer

This version is intentionally still algorithmic, not ML.
Compared with the previous grouping-aware shelf baseline, v8 adds:

1. Hard-constraint safety first
   - preplaced blocks keep exact x/y/w/h
   - fixed-shape blocks keep exact w/h
   - soft blocks keep area exactly, unless a safe MIB unification is possible

2. Better MIB handling
   - if a MIB group has a fixed/preplaced master and movable soft blocks with
     compatible area, copy the master shape
   - otherwise only unify shapes when target areas are within 1% tolerance

3. Compact grouping items
   - non-preplaced cluster/group members are packed as compact 2D composite items
   - candidates include horizontal, vertical, two-row, and near-square grid layouts

4. Preplaced-aware grouping
   - when a group contains preplaced blocks, movable group members are first tried
     adjacent to the preplaced/member rectangles to reduce grouping fragmentation

5. Boundary-aware frame packing
   - LEFT=1, RIGHT=2, TOP=4, BOTTOM=8 are treated as real placement constraints
   - boundary items are placed on the requested frame edge/corner when possible

6. Obstacle-aware placement
   - preplaced blocks are obstacles, not a reason to push everything to the right
   - remaining items are inserted into a fixed frame by bottom-left style candidates
   - if the frame is too tight, the solver expands and retries

The goal of v8 is to keep feasibility while reducing soft-constraint violations first.
"""

import math


LEFT = 1
RIGHT = 2
TOP = 4
BOTTOM = 8
EPS = 1e-9
MARGIN = 1e-4


class MyOptimizer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    # ================================================================
    # Public API expected by the contest framework
    # ================================================================
    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ):
        positions = [None] * block_count
        dims = [None] * block_count

        fixed_indices = set()
        preplaced_indices = set()

        # ------------------------------------------------------------
        # 1. Decide initial dimensions.
        # ------------------------------------------------------------
        for i in range(block_count):
            is_fixed = self._is_fixed(i, constraints)
            is_preplaced = self._is_preplaced(i, constraints)

            if is_fixed:
                fixed_indices.add(i)
            if is_preplaced:
                preplaced_indices.add(i)

            if is_preplaced and target_positions is not None:
                x, y, w, h = self._target_xywh(i, target_positions)
                positions[i] = (x, y, w, h)
                dims[i] = (w, h)

            elif is_fixed and target_positions is not None:
                _, _, w, h = self._target_xywh(i, target_positions)
                if w <= 0 or h <= 0:
                    area = self._area(i, area_targets)
                    side = math.sqrt(area)
                    w, h = side, side
                dims[i] = (w, h)

            else:
                area = self._area(i, area_targets)
                code = self._boundary_code(i, constraints)
                dims[i] = self._default_soft_dim(area, code)

        # ------------------------------------------------------------
        # 2. MIB-aware safe dimension unification.
        # ------------------------------------------------------------
        self._apply_safe_mib_dimensions(
            block_count=block_count,
            area_targets=area_targets,
            constraints=constraints,
            dims=dims,
            fixed_indices=fixed_indices,
            preplaced_indices=preplaced_indices,
        )

        # ------------------------------------------------------------
        # 3. Build compact placement items.
        # ------------------------------------------------------------
        anchored_clusters, items = self._build_items(
            block_count=block_count,
            dims=dims,
            constraints=constraints,
            fixed_indices=fixed_indices,
            preplaced_indices=preplaced_indices,
        )

        # ------------------------------------------------------------
        # 4. Connectivity/pin anchor estimation.
        #    This is deliberately light-weight; it only guides greedy tie-breaks.
        # ------------------------------------------------------------
        block_anchors = self._estimate_block_anchors(
            block_count=block_count,
            positions=positions,
            dims=dims,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
        )
        for item in items:
            item["anchor"] = self._item_anchor(item, block_anchors, dims)

        # ------------------------------------------------------------
        # 5. Try obstacle-aware frame packing.
        # ------------------------------------------------------------
        best_positions = None
        best_score = None

        frame_candidates = self._frame_candidates(block_count, dims, positions, items)
        successful_trials = 0
        # Large instances dominate the final score, but runtime still matters.
        # Try enough frames to avoid an obviously bad outline, then stop early.
        if block_count >= 100:
            max_successful_trials = 2
        elif block_count >= 60:
            max_successful_trials = 3
        else:
            max_successful_trials = 5

        for frame_w, frame_h in frame_candidates:
            cand = self._pack_in_frame(
                block_count=block_count,
                base_positions=positions,
                dims=dims,
                constraints=constraints,
                items=items,
                anchored_clusters=anchored_clusters,
                block_anchors=block_anchors,
                frame_w=frame_w,
                frame_h=frame_h,
            )

            if cand is None:
                continue

            successful_trials += 1
            cand = self._final_boundary_nudge(cand, dims, constraints)
            score = self._layout_score(cand, constraints, b2b_connectivity, p2b_connectivity, pins_pos)

            if best_score is None or score < best_score:
                best_score = score
                best_positions = cand

            if successful_trials >= max_successful_trials:
                break

        # ------------------------------------------------------------
        # 6. Robust fallback: old safe shelf packing to the right of all preplaced blocks.
        #    This should rarely be used, but it protects feasibility.
        # ------------------------------------------------------------
        if best_positions is None:
            best_positions = self._safe_shelf_fallback(
                block_count=block_count,
                positions=positions,
                dims=dims,
                items=items,
                constraints=constraints,
            )

        return best_positions

    # ================================================================
    # Dimension helpers
    # ================================================================
    def _apply_safe_mib_dimensions(
        self,
        block_count,
        area_targets,
        constraints,
        dims,
        fixed_indices,
        preplaced_indices,
    ):
        if constraints is None or self._num_constraint_cols(constraints) <= 2:
            return

        groups = {}
        for i in range(block_count):
            gid = self._constraint_int(constraints, i, 2)
            if gid > 0:
                groups.setdefault(gid, []).append(i)

        for _, members in groups.items():
            if len(members) <= 1:
                continue

            # Prefer a fixed/preplaced shape as the master when it is area-compatible.
            master = None
            for i in members:
                if i in fixed_indices or i in preplaced_indices:
                    master = i
                    break

            if master is not None:
                mw, mh = dims[master]
                master_area = mw * mh
                ok = True
                for i in members:
                    if i in fixed_indices or i in preplaced_indices:
                        continue
                    target_area = self._area(i, area_targets)
                    if target_area <= 0 or abs(master_area - target_area) / target_area > 0.01:
                        ok = False
                        break
                if ok:
                    for i in members:
                        if i not in fixed_indices and i not in preplaced_indices:
                            dims[i] = (mw, mh)
                    continue

            # Otherwise unify only if all movable target areas are mutually compatible.
            movable = [i for i in members if i not in fixed_indices and i not in preplaced_indices]
            if len(movable) <= 1:
                continue

            areas = [self._area(i, area_targets) for i in movable]
            if any(a <= 0 for a in areas):
                continue

            avg_area = sum(areas) / len(areas)
            if all(abs(avg_area - a) / a <= 0.01 for a in areas):
                side = math.sqrt(avg_area)
                for i in movable:
                    dims[i] = (side, side)

    def _default_soft_dim(self, area, boundary_code):
        # Gentle aspect-ratio bias. Keep exact area for hard feasibility.
        if boundary_code & (LEFT | RIGHT) and not (boundary_code & (TOP | BOTTOM)):
            ratio = 0.75  # narrower/taller edge block
        elif boundary_code & (TOP | BOTTOM) and not (boundary_code & (LEFT | RIGHT)):
            ratio = 1.33  # wider/shorter edge block
        else:
            ratio = 1.0

        w = math.sqrt(area * ratio)
        h = area / w
        return (w, h)

    # ================================================================
    # Item construction
    # ================================================================
    def _build_items(self, block_count, dims, constraints, fixed_indices, preplaced_indices):
        used = set()
        items = []
        anchored_clusters = {}

        if constraints is not None and self._num_constraint_cols(constraints) > 3:
            cluster_map = {}
            for i in range(block_count):
                cid = self._constraint_int(constraints, i, 3)
                if cid > 0:
                    cluster_map.setdefault(cid, []).append(i)

            for cid in sorted(cluster_map.keys()):
                members_all = cluster_map[cid]
                movable = [i for i in members_all if i not in preplaced_indices]
                preplaced = [i for i in members_all if i in preplaced_indices]

                if preplaced and movable:
                    anchored_clusters[cid] = {
                        "preplaced": preplaced,
                        "movable": movable,
                    }
                    # These movable blocks will first be tried near their preplaced anchor.
                    # If that fails, they will be added as singles below.
                    continue

                if len(movable) <= 1:
                    continue

                item = self._make_compact_group_item(movable, dims, constraints)
                items.append(item)
                for b in movable:
                    used.add(b)

        # Remaining non-preplaced blocks are single items.
        for i in range(block_count):
            if i in preplaced_indices or i in used:
                continue
            w, h = dims[i]
            items.append({
                "blocks": [i],
                "w": w,
                "h": h,
                "offsets": {i: (0.0, 0.0)},
                "kind": "single",
                "boundary_score": self._block_boundary_score(i, constraints),
            })

        # Sort: boundary first, then larger/more connected-ish compact items.
        items.sort(
            key=lambda it: (
                self._item_boundary_score(it, constraints),
                len(it["blocks"]),
                it["w"] * it["h"],
                max(it["w"], it["h"]),
            ),
            reverse=True,
        )
        return anchored_clusters, items

    def _make_compact_group_item(self, members, dims, constraints):
        # Boundary members first so a boundary-constrained block has a chance to sit on
        # the outer edge of the composite item.
        members = sorted(
            members,
            key=lambda i: (
                self._block_boundary_score(i, constraints),
                dims[i][0] * dims[i][1],
                max(dims[i]),
            ),
            reverse=True,
        )

        candidates = []
        candidates.append(self._layout_group_horizontal(members, dims))
        candidates.append(self._layout_group_vertical(members, dims))
        if len(members) >= 3:
            candidates.append(self._layout_group_two_rows(members, dims))
            candidates.append(self._layout_group_grid(members, dims))

        # Choose the most compact bbox, with small tie-breaker for squareness.
        best = min(
            candidates,
            key=lambda c: (c["w"] * c["h"], abs(c["w"] - c["h"]), max(c["w"], c["h"])),
        )
        best["kind"] = "group"
        best["boundary_score"] = self._item_boundary_score(best, constraints)
        return best

    def _layout_group_horizontal(self, members, dims):
        offsets = {}
        x = 0.0
        max_h = 0.0
        for i in members:
            offsets[i] = (x, 0.0)
            w, h = dims[i]
            x += w
            max_h = max(max_h, h)
        return {"blocks": members, "w": x, "h": max_h, "offsets": offsets}

    def _layout_group_vertical(self, members, dims):
        offsets = {}
        y = 0.0
        max_w = 0.0
        for i in members:
            offsets[i] = (0.0, y)
            w, h = dims[i]
            y += h
            max_w = max(max_w, w)
        return {"blocks": members, "w": max_w, "h": y, "offsets": offsets}

    def _layout_group_two_rows(self, members, dims):
        # Greedy balance by width into two touching rows.
        row1, row2 = [], []
        w1 = w2 = 0.0
        for i in members:
            if w1 <= w2:
                row1.append(i)
                w1 += dims[i][0]
            else:
                row2.append(i)
                w2 += dims[i][0]

        offsets = {}
        x = 0.0
        row1_h = max((dims[i][1] for i in row1), default=0.0)
        row2_h = max((dims[i][1] for i in row2), default=0.0)

        for i in row1:
            offsets[i] = (x, 0.0)
            x += dims[i][0]
        x = 0.0
        for i in row2:
            offsets[i] = (x, row1_h)
            x += dims[i][0]

        return {
            "blocks": members,
            "w": max(w1, w2),
            "h": row1_h + row2_h,
            "offsets": offsets,
        }

    def _layout_group_grid(self, members, dims):
        # Simple near-square row packing. Rows touch vertically; blocks touch horizontally.
        total_area = sum(dims[i][0] * dims[i][1] for i in members)
        target_w = math.sqrt(max(total_area, EPS))

        offsets = {}
        rows = []
        cur = []
        cur_w = 0.0
        cur_h = 0.0

        for i in members:
            w, h = dims[i]
            if cur and cur_w + w > target_w:
                rows.append((cur, cur_w, cur_h))
                cur = []
                cur_w = 0.0
                cur_h = 0.0
            cur.append(i)
            cur_w += w
            cur_h = max(cur_h, h)
        if cur:
            rows.append((cur, cur_w, cur_h))

        y = 0.0
        max_w = 0.0
        for row, row_w, row_h in rows:
            x = 0.0
            for i in row:
                offsets[i] = (x, y)
                x += dims[i][0]
            y += row_h
            max_w = max(max_w, row_w)

        return {"blocks": members, "w": max_w, "h": y, "offsets": offsets}

    # ================================================================
    # Frame packing
    # ================================================================
    def _frame_candidates(self, block_count, dims, positions, items):
        total_area = sum(max(dims[i][0] * dims[i][1], 0.0) for i in range(block_count))
        base = math.sqrt(max(total_area, 1.0))

        pre_w = max((p[0] + p[2] for p in positions if p is not None), default=0.0)
        pre_h = max((p[1] + p[3] for p in positions if p is not None), default=0.0)
        max_item_w = max((it["w"] for it in items), default=1.0)
        max_item_h = max((it["h"] for it in items), default=1.0)

        # Keep the candidate set compact. The evaluator rewards runtime too, and
        # a few good fixed-outline retries are usually enough for this greedy placer.
        aspects = [1.0, 1.35, 0.75, 1.8, 0.55]
        scales = [1.05, 1.15, 1.35, 1.65, 2.10]

        seen = set()
        frames = []
        for scale in scales:
            for aspect in aspects:
                w = base * scale * math.sqrt(aspect)
                h = base * scale / math.sqrt(aspect)
                w = max(w, pre_w + MARGIN, max_item_w + MARGIN)
                h = max(h, pre_h + MARGIN, max_item_h + MARGIN)
                key = (round(w, 6), round(h, 6))
                if key not in seen:
                    seen.add(key)
                    frames.append((w, h))

        # Smaller area frames first.
        frames.sort(key=lambda wh: (wh[0] * wh[1], max(wh[0], wh[1])))
        return frames

    def _pack_in_frame(
        self,
        block_count,
        base_positions,
        dims,
        constraints,
        items,
        anchored_clusters,
        block_anchors,
        frame_w,
        frame_h,
    ):
        positions = list(base_positions)
        placed_rects = []
        placed_blocks = set()

        # Add preplaced obstacles.
        for i, p in enumerate(positions):
            if p is not None:
                if not self._inside_frame(p[0], p[1], p[2], p[3], frame_w, frame_h):
                    return None
                if self._overlaps_any(p, placed_rects):
                    return None
                placed_rects.append((p[0], p[1], p[2], p[3], i))
                placed_blocks.add(i)

        # First try to attach movable members to preplaced clusters.
        anchored_done = set()
        for _, data in anchored_clusters.items():
            cluster_rects = []
            for b in data["preplaced"]:
                if positions[b] is not None:
                    x, y, w, h = positions[b]
                    cluster_rects.append((x, y, w, h, b))

            movables = sorted(
                data["movable"],
                key=lambda b: (
                    self._block_boundary_score(b, constraints),
                    dims[b][0] * dims[b][1],
                ),
                reverse=True,
            )

            for b in movables:
                if b in placed_blocks:
                    continue
                bw, bh = dims[b]
                candidates = self._adjacent_candidates_for_block(
                    bw, bh, cluster_rects, frame_w, frame_h, self._boundary_code(b, constraints)
                )
                best = None
                best_score = None
                for x, y in candidates:
                    rect = (x, y, bw, bh, b)
                    if not self._inside_frame(x, y, bw, bh, frame_w, frame_h):
                        continue
                    if self._overlaps_any(rect, placed_rects):
                        continue
                    score = self._candidate_score_block(
                        x, y, bw, bh, b, positions, placed_rects, frame_w, frame_h, block_anchors, constraints
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best = rect

                if best is not None:
                    x, y, bw, bh, _ = best
                    positions[b] = (x, y, bw, bh)
                    placed_rects.append(best)
                    cluster_rects.append(best)
                    placed_blocks.add(b)
                    anchored_done.add(b)

        # Place all remaining items.
        for item in items:
            # Some items may have blocks already attached to preplaced clusters.
            if all(b in placed_blocks for b in item["blocks"]):
                continue

            # If this is a multi-block group and part of it is already placed,
            # skip the composite and let unplaced members be handled by fallback singles.
            if any(b in placed_blocks for b in item["blocks"]):
                continue

            candidates = self._item_candidate_positions(item, placed_rects, frame_w, frame_h, constraints)
            best_pos = None
            best_score = None

            for x, y in candidates:
                if not self._inside_frame(x, y, item["w"], item["h"], frame_w, frame_h):
                    continue
                item_rects = self._item_rects_at(item, dims, x, y)
                if any(self._overlaps_any(r, placed_rects) for r in item_rects):
                    continue

                score = self._candidate_score_item(
                    item, x, y, positions, placed_rects, frame_w, frame_h, constraints
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (x, y)

            if best_pos is None:
                return None

            x, y = best_pos
            for b in item["blocks"]:
                ox, oy = item["offsets"][b]
                bw, bh = dims[b]
                positions[b] = (x + ox, y + oy, bw, bh)
                placed_rects.append((x + ox, y + oy, bw, bh, b))
                placed_blocks.add(b)

        # Any remaining unplaced block from an anchored cluster gets packed as a single.
        for b in range(block_count):
            if positions[b] is not None:
                continue
            bw, bh = dims[b]
            single = {
                "blocks": [b],
                "w": bw,
                "h": bh,
                "offsets": {b: (0.0, 0.0)},
                "kind": "late_single",
            }
            candidates = self._item_candidate_positions(single, placed_rects, frame_w, frame_h, constraints)
            best_pos = None
            best_score = None
            for x, y in candidates:
                rect = (x, y, bw, bh, b)
                if not self._inside_frame(x, y, bw, bh, frame_w, frame_h):
                    continue
                if self._overlaps_any(rect, placed_rects):
                    continue
                score = self._candidate_score_block(
                    x, y, bw, bh, b, positions, placed_rects, frame_w, frame_h, block_anchors, constraints
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (x, y)
            if best_pos is None:
                return None
            x, y = best_pos
            positions[b] = (x, y, bw, bh)
            placed_rects.append((x, y, bw, bh, b))

        return positions

    def _item_candidate_positions(self, item, placed_rects, frame_w, frame_h, constraints):
        # Candidate count must stay small for 100+ block cases.  Instead of
        # trying every x-edge times every y-edge, use standard bottom-left
        # corner candidates around each placed obstacle.
        candidates = [
            (0.0, 0.0),
            (max(0.0, frame_w - item["w"]), 0.0),
            (0.0, max(0.0, frame_h - item["h"])),
            (max(0.0, frame_w - item["w"]), max(0.0, frame_h - item["h"])),
        ]

        xs = {0.0, max(0.0, frame_w - item["w"])}
        ys = {0.0, max(0.0, frame_h - item["h"])}

        for rx, ry, rw, rh, _ in placed_rects:
            right_x = rx + rw + MARGIN
            top_y = ry + rh + MARGIN
            left_x = max(0.0, rx - item["w"] - MARGIN)
            below_y = max(0.0, ry - item["h"] - MARGIN)

            xs.update([right_x, left_x, rx])
            ys.update([top_y, below_y, ry])

            candidates.extend([
                (right_x, ry),
                (right_x, max(0.0, ry + rh - item["h"])),
                (rx, top_y),
                (max(0.0, rx + rw - item["w"]), top_y),
                (left_x, ry),
                (left_x, max(0.0, ry + rh - item["h"])),
                (rx, below_y),
                (max(0.0, rx + rw - item["w"]), below_y),
            ])

        boundary_positions = self._boundary_positions_for_item(item, frame_w, frame_h, constraints)
        if boundary_positions:
            for bx, by in boundary_positions:
                candidates.append((bx, by))

                exact_left_or_right = (
                    abs(bx - 0.0) <= 1e-9
                    or abs(bx - max(0.0, frame_w - item["w"])) <= 1e-9
                )
                exact_top_or_bottom = (
                    abs(by - 0.0) <= 1e-9
                    or abs(by - max(0.0, frame_h - item["h"])) <= 1e-9
                )

                # If one axis is fixed by boundary, slide along the other axis.
                if exact_left_or_right:
                    for y in ys:
                        candidates.append((bx, y))
                if exact_top_or_bottom:
                    for x in xs:
                        candidates.append((x, by))

        cleaned = []
        seen = set()
        for x, y in candidates:
            x = min(max(0.0, x), max(0.0, frame_w - item["w"]))
            y = min(max(0.0, y), max(0.0, frame_h - item["h"]))
            key = (round(x, 7), round(y, 7))
            if key not in seen:
                seen.add(key)
                cleaned.append((x, y))

        cleaned.sort(key=lambda p: (p[1], p[0]))
        return cleaned

    def _boundary_positions_for_item(self, item, frame_w, frame_h, constraints):
        if constraints is None or self._num_constraint_cols(constraints) <= 4:
            return []

        x_values = []
        y_values = []
        has_boundary = False

        for b in item["blocks"]:
            code = self._boundary_code(b, constraints)
            if code == 0:
                continue
            has_boundary = True
            ox, oy = item["offsets"].get(b, (0.0, 0.0))
            bw = item["w"] if b not in item["offsets"] else None
            # Actual block dimension is not stored here directly; use item extents for
            # fallback and exact offsets for common cases below.
            # The exact width/height are recovered from item rectangle geometry later.
            if code & LEFT:
                x_values.append(-ox)
            if code & RIGHT:
                # item['w'] - (ox + block_w) is unknown here without dims.
                # Using item right edge is exact for singles and often good for compact groups.
                x_values.append(frame_w - item["w"])
            if code & BOTTOM:
                y_values.append(-oy)
            if code & TOP:
                y_values.append(frame_h - item["h"])

        if not has_boundary:
            return []

        if not x_values:
            x_values = [0.0, max(0.0, frame_w - item["w"])]
        if not y_values:
            y_values = [0.0, max(0.0, frame_h - item["h"])]

        out = []
        for x in x_values:
            for y in y_values:
                out.append((x, y))
        return out

    def _adjacent_candidates_for_block(self, w, h, cluster_rects, frame_w, frame_h, code):
        candidates = []
        for rx, ry, rw, rh, _ in cluster_rects:
            # Right / left of an existing group rect.
            candidates.append((rx + rw, ry))
            candidates.append((rx + rw, max(0.0, ry + rh - h)))
            candidates.append((rx - w, ry))
            candidates.append((rx - w, max(0.0, ry + rh - h)))
            # Above / below an existing group rect.
            candidates.append((rx, ry + rh))
            candidates.append((max(0.0, rx + rw - w), ry + rh))
            candidates.append((rx, ry - h))
            candidates.append((max(0.0, rx + rw - w), ry - h))

        # Exact frame boundary candidates if the block has boundary constraint.
        xs = []
        ys = []
        if code & LEFT:
            xs.append(0.0)
        if code & RIGHT:
            xs.append(frame_w - w)
        if code & BOTTOM:
            ys.append(0.0)
        if code & TOP:
            ys.append(frame_h - h)
        if xs or ys:
            if not xs:
                xs = [x for x, _ in candidates] + [0.0, frame_w - w]
            if not ys:
                ys = [y for _, y in candidates] + [0.0, frame_h - h]
            for x in xs:
                for y in ys:
                    candidates.append((x, y))

        cleaned = []
        seen = set()
        for x, y in candidates:
            x = min(max(0.0, x), max(0.0, frame_w - w))
            y = min(max(0.0, y), max(0.0, frame_h - h))
            key = (round(x, 7), round(y, 7))
            if key not in seen:
                seen.add(key)
                cleaned.append((x, y))
        cleaned.sort(key=lambda p: (p[1], p[0]))
        return cleaned

    # ================================================================
    # Scoring / refinement helpers
    # ================================================================
    def _candidate_score_item(self, item, x, y, positions, placed_rects, frame_w, frame_h, constraints):
        cx = x + item["w"] / 2.0
        cy = y + item["h"] / 2.0
        ax, ay, aw = item.get("anchor", (frame_w / 2.0, frame_h / 2.0, 0.0))
        anchor_dist = 0.0 if aw <= 0 else abs(cx - ax) + abs(cy - ay)

        # Prefer lower-left compaction, but make boundary satisfaction much more important.
        boundary_penalty = self._item_boundary_penalty_est(item, x, y, frame_w, frame_h, constraints)
        bbox_area = self._bbox_area_with_rect(positions, (x, y, item["w"], item["h"]))
        return bbox_area + 0.08 * anchor_dist + 1000.0 * boundary_penalty + 1e-3 * y + 1e-4 * x

    def _candidate_score_block(self, x, y, w, h, b, positions, placed_rects, frame_w, frame_h, block_anchors, constraints):
        cx = x + w / 2.0
        cy = y + h / 2.0
        ax, ay, aw = block_anchors[b]
        anchor_dist = 0.0 if aw <= 0 else abs(cx - ax) + abs(cy - ay)
        boundary_penalty = self._block_boundary_penalty_est(b, x, y, w, h, frame_w, frame_h, constraints)
        bbox_area = self._bbox_area_with_rect(positions, (x, y, w, h))
        return bbox_area + 0.08 * anchor_dist + 1000.0 * boundary_penalty + 1e-3 * y + 1e-4 * x

    def _layout_score(self, positions, constraints, b2b_connectivity, p2b_connectivity, pins_pos):
        # Fast approximate objective for choosing among candidate layouts.
        bbox_area = self._bbox_area(positions)
        boundary_v = self._count_boundary_violations(positions, constraints)
        group_v = self._count_group_fragments(positions, constraints)
        hpwl = self._approx_hpwl(positions, b2b_connectivity, p2b_connectivity, pins_pos)
        return bbox_area + 0.05 * hpwl + 5000.0 * boundary_v + 3000.0 * group_v

    def _final_boundary_nudge(self, positions, dims, constraints):
        if constraints is None or self._num_constraint_cols(constraints) <= 4:
            return positions

        out = list(positions)
        xmin, ymin, xmax, ymax = self._bbox(out)

        # Only nudge non-cluster single blocks. Moving group blocks alone would destroy grouping.
        for i, p in enumerate(list(out)):
            code = self._boundary_code(i, constraints)
            if code == 0 or self._is_preplaced(i, constraints):
                continue
            if self._constraint_int(constraints, i, 3) > 0:
                continue

            x, y, w, h = p
            xs = [x]
            ys = [y]
            if code & LEFT:
                xs = [xmin]
            if code & RIGHT:
                xs = [xmax - w]
            if code & BOTTOM:
                ys = [ymin]
            if code & TOP:
                ys = [ymax - h]

            moved = False
            for nx in xs:
                for ny in ys:
                    rect = (nx, ny, w, h, i)
                    others = [
                        (q[0], q[1], q[2], q[3], j)
                        for j, q in enumerate(out)
                        if j != i and q is not None
                    ]
                    if not self._overlaps_any(rect, others):
                        out[i] = (nx, ny, w, h)
                        moved = True
                        break
                if moved:
                    break
        return out

    # ================================================================
    # Fallback
    # ================================================================
    def _safe_shelf_fallback(self, block_count, positions, dims, items, constraints):
        preplaced_xmax = max((p[0] + p[2] for p in positions if p is not None), default=0.0)
        preplaced_ymin = min((p[1] for p in positions if p is not None), default=0.0)

        remaining_items = []
        placed = {i for i, p in enumerate(positions) if p is not None}
        for item in items:
            if any(b in placed for b in item["blocks"]):
                continue
            remaining_items.append(item)

        total_item_area = sum(it["w"] * it["h"] for it in remaining_items)
        max_item_width = max((it["w"] for it in remaining_items), default=1.0)
        base_width = math.sqrt(max(total_item_area, 1.0))
        shelf_width = max(max_item_width, base_width * 1.25)

        x_start = preplaced_xmax + MARGIN
        y_start = preplaced_ymin
        x_cursor = x_start
        y_cursor = y_start
        row_height = 0.0

        out = list(positions)
        for item in remaining_items:
            if x_cursor > x_start and x_cursor + item["w"] > x_start + shelf_width:
                y_cursor += row_height + MARGIN
                x_cursor = x_start
                row_height = 0.0

            for b in item["blocks"]:
                ox, oy = item["offsets"][b]
                bw, bh = dims[b]
                out[b] = (x_cursor + ox, y_cursor + oy, bw, bh)
            x_cursor += item["w"] + MARGIN
            row_height = max(row_height, item["h"])

        # Last safety net.
        x_cursor = preplaced_xmax + MARGIN
        for i in range(block_count):
            if out[i] is None:
                bw, bh = dims[i]
                out[i] = (x_cursor, 0.0, bw, bh)
                x_cursor += bw + MARGIN

        return out

    # ================================================================
    # Geometry helpers
    # ================================================================
    def _item_rects_at(self, item, dims, x, y):
        rects = []
        for b in item["blocks"]:
            ox, oy = item["offsets"][b]
            bw, bh = dims[b]
            rects.append((x + ox, y + oy, bw, bh, b))
        return rects

    def _inside_frame(self, x, y, w, h, frame_w, frame_h):
        return x >= -EPS and y >= -EPS and x + w <= frame_w + EPS and y + h <= frame_h + EPS

    def _overlaps_any(self, rect, rects):
        x, y, w, h = rect[:4]
        for r in rects:
            rx, ry, rw, rh = r[:4]
            if self._rect_overlap(x, y, w, h, rx, ry, rw, rh):
                return True
        return False

    def _rect_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return not (
            x1 + w1 <= x2 + EPS
            or x2 + w2 <= x1 + EPS
            or y1 + h1 <= y2 + EPS
            or y2 + h2 <= y1 + EPS
        )

    def _bbox(self, positions):
        valid = [p for p in positions if p is not None]
        if not valid:
            return (0.0, 0.0, 1.0, 1.0)
        xmin = min(p[0] for p in valid)
        ymin = min(p[1] for p in valid)
        xmax = max(p[0] + p[2] for p in valid)
        ymax = max(p[1] + p[3] for p in valid)
        return xmin, ymin, xmax, ymax

    def _bbox_area(self, positions):
        xmin, ymin, xmax, ymax = self._bbox(positions)
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

    def _bbox_area_with_rect(self, positions, rect):
        valid = [p for p in positions if p is not None]
        valid.append(rect[:4])
        xmin = min(p[0] for p in valid)
        ymin = min(p[1] for p in valid)
        xmax = max(p[0] + p[2] for p in valid)
        ymax = max(p[1] + p[3] for p in valid)
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

    # ================================================================
    # Constraint / violation estimation
    # ================================================================
    def _item_boundary_score(self, item, constraints):
        return sum(self._block_boundary_score(b, constraints) for b in item["blocks"])

    def _block_boundary_score(self, b, constraints):
        code = self._boundary_code(b, constraints)
        if code == 0:
            return 0
        score = 10
        for bit in (LEFT, RIGHT, TOP, BOTTOM):
            if code & bit:
                score += 1
        return score

    def _item_boundary_penalty_est(self, item, x, y, frame_w, frame_h, constraints):
        penalty = 0
        for b in item["blocks"]:
            ox, oy = item["offsets"][b]
            # Approximate with item boundary when dims are not passed in this helper.
            bx = x + ox
            by = y + oy
            bw = item["w"]
            bh = item["h"]
            penalty += self._block_boundary_penalty_est(b, bx, by, bw, bh, frame_w, frame_h, constraints)
        return penalty

    def _block_boundary_penalty_est(self, b, x, y, w, h, frame_w, frame_h, constraints):
        code = self._boundary_code(b, constraints)
        if code == 0:
            return 0
        bad = 0
        if code & LEFT and abs(x - 0.0) > 1e-6:
            bad += 1
        if code & RIGHT and abs((x + w) - frame_w) > 1e-6:
            bad += 1
        if code & TOP and abs((y + h) - frame_h) > 1e-6:
            bad += 1
        if code & BOTTOM and abs(y - 0.0) > 1e-6:
            bad += 1
        return bad

    def _count_boundary_violations(self, positions, constraints):
        if constraints is None or self._num_constraint_cols(constraints) <= 4:
            return 0
        xmin, ymin, xmax, ymax = self._bbox(positions)
        bad = 0
        for i, p in enumerate(positions):
            code = self._boundary_code(i, constraints)
            if code == 0 or p is None:
                continue
            x, y, w, h = p
            ok = True
            if code & LEFT:
                ok = ok and abs(x - xmin) <= 1e-6
            if code & RIGHT:
                ok = ok and abs(x + w - xmax) <= 1e-6
            if code & TOP:
                ok = ok and abs(y + h - ymax) <= 1e-6
            if code & BOTTOM:
                ok = ok and abs(y - ymin) <= 1e-6
            if not ok:
                bad += 1
        return bad

    def _count_group_fragments(self, positions, constraints):
        if constraints is None or self._num_constraint_cols(constraints) <= 3:
            return 0
        groups = {}
        for i, p in enumerate(positions):
            cid = self._constraint_int(constraints, i, 3)
            if cid > 0 and p is not None:
                groups.setdefault(cid, []).append(i)

        total_frag = 0
        for members in groups.values():
            if len(members) <= 1:
                continue
            comps = self._connected_components_by_touch(members, positions)
            total_frag += max(0, comps - 1)
        return total_frag

    def _connected_components_by_touch(self, members, positions):
        parent = {i: i for i in members}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for idx, a in enumerate(members):
            for b in members[idx + 1:]:
                if self._rect_touch_edge(positions[a], positions[b]):
                    union(a, b)
        return len({find(i) for i in members})

    def _rect_touch_edge(self, p, q):
        x1, y1, w1, h1 = p
        x2, y2, w2, h2 = q
        # vertical shared edge with non-zero y overlap
        if abs((x1 + w1) - x2) <= 1e-6 or abs((x2 + w2) - x1) <= 1e-6:
            overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            if overlap > 1e-6:
                return True
        # horizontal shared edge with non-zero x overlap
        if abs((y1 + h1) - y2) <= 1e-6 or abs((y2 + h2) - y1) <= 1e-6:
            overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            if overlap > 1e-6:
                return True
        return False

    # ================================================================
    # Lightweight connectivity anchors / HPWL estimate
    # ================================================================
    def _estimate_block_anchors(self, block_count, positions, dims, b2b_connectivity, p2b_connectivity, pins_pos):
        sums = [[0.0, 0.0, 0.0] for _ in range(block_count)]

        # Preplaced connected blocks as anchors.
        if self._looks_square_matrix(b2b_connectivity, block_count):
            for i in range(block_count):
                for j in range(block_count):
                    if i == j:
                        continue
                    w = self._matrix_val(b2b_connectivity, i, j)
                    if w <= 0:
                        continue
                    if positions[j] is not None:
                        cx = positions[j][0] + positions[j][2] / 2.0
                        cy = positions[j][1] + positions[j][3] / 2.0
                        sums[i][0] += w * cx
                        sums[i][1] += w * cy
                        sums[i][2] += w

        # Pin / terminal anchors. Try both [block, pin] and [pin, block] conventions.
        pin_list = self._pin_list(pins_pos)
        if pin_list and p2b_connectivity is not None:
            rows, cols = self._shape2(p2b_connectivity)
            if rows == block_count:
                for i in range(block_count):
                    for pidx, (px, py) in enumerate(pin_list[:cols]):
                        w = self._matrix_val(p2b_connectivity, i, pidx)
                        if w > 0:
                            sums[i][0] += w * px
                            sums[i][1] += w * py
                            sums[i][2] += w
            elif cols == block_count:
                for pidx, (px, py) in enumerate(pin_list[:rows]):
                    for i in range(block_count):
                        w = self._matrix_val(p2b_connectivity, pidx, i)
                        if w > 0:
                            sums[i][0] += w * px
                            sums[i][1] += w * py
                            sums[i][2] += w

        anchors = []
        for i in range(block_count):
            sx, sy, sw = sums[i]
            if sw > 0:
                anchors.append((sx / sw, sy / sw, sw))
            else:
                anchors.append((0.0, 0.0, 0.0))
        return anchors

    def _item_anchor(self, item, block_anchors, dims):
        sx = sy = sw = 0.0
        for b in item["blocks"]:
            ax, ay, aw = block_anchors[b]
            if aw <= 0:
                continue
            area = dims[b][0] * dims[b][1]
            weight = aw * max(area, EPS)
            sx += weight * ax
            sy += weight * ay
            sw += weight
        if sw <= 0:
            return (0.0, 0.0, 0.0)
        return (sx / sw, sy / sw, sw)

    def _approx_hpwl(self, positions, b2b_connectivity, p2b_connectivity, pins_pos):
        hpwl = 0.0
        n = len(positions)
        centers = [(p[0] + p[2] / 2.0, p[1] + p[3] / 2.0) for p in positions]

        if self._looks_square_matrix(b2b_connectivity, n):
            for i in range(n):
                for j in range(i + 1, n):
                    w = self._matrix_val(b2b_connectivity, i, j)
                    if w > 0:
                        hpwl += w * (abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1]))

        pin_list = self._pin_list(pins_pos)
        if pin_list and p2b_connectivity is not None:
            rows, cols = self._shape2(p2b_connectivity)
            if rows == n:
                for i in range(n):
                    for pidx, (px, py) in enumerate(pin_list[:cols]):
                        w = self._matrix_val(p2b_connectivity, i, pidx)
                        if w > 0:
                            hpwl += w * (abs(centers[i][0] - px) + abs(centers[i][1] - py))
            elif cols == n:
                for pidx, (px, py) in enumerate(pin_list[:rows]):
                    for i in range(n):
                        w = self._matrix_val(p2b_connectivity, pidx, i)
                        if w > 0:
                            hpwl += w * (abs(centers[i][0] - px) + abs(centers[i][1] - py))
        return hpwl

    # ================================================================
    # Generic tensor / numpy / list access helpers
    # ================================================================
    def _area(self, i, area_targets):
        a = self._to_float(area_targets[i])
        return a if a > 0 else 1.0

    def _target_xywh(self, i, target_positions):
        row = target_positions[i]
        return (
            self._to_float(row[0]),
            self._to_float(row[1]),
            self._to_float(row[2]),
            self._to_float(row[3]),
        )

    def _is_fixed(self, i, constraints):
        return constraints is not None and self._num_constraint_cols(constraints) > 0 and self._constraint_int(constraints, i, 0) != 0

    def _is_preplaced(self, i, constraints):
        return constraints is not None and self._num_constraint_cols(constraints) > 1 and self._constraint_int(constraints, i, 1) != 0

    def _boundary_code(self, i, constraints):
        if constraints is None or self._num_constraint_cols(constraints) <= 4:
            return 0
        return self._constraint_int(constraints, i, 4)

    def _constraint_int(self, constraints, i, j):
        try:
            return int(self._to_float(constraints[i, j]))
        except Exception:
            return int(self._to_float(constraints[i][j]))

    def _num_constraint_cols(self, constraints):
        try:
            return int(constraints.shape[1])
        except Exception:
            if constraints is None or len(constraints) == 0:
                return 0
            return len(constraints[0])

    def _to_float(self, x):
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)

    def _shape2(self, x):
        if x is None:
            return (0, 0)
        try:
            s = x.shape
            if len(s) >= 2:
                return int(s[0]), int(s[1])
        except Exception:
            pass
        try:
            return len(x), len(x[0]) if len(x) > 0 else 0
        except Exception:
            return (0, 0)

    def _looks_square_matrix(self, x, n):
        rows, cols = self._shape2(x)
        return rows == n and cols == n

    def _matrix_val(self, mat, i, j):
        try:
            return self._to_float(mat[i, j])
        except Exception:
            try:
                return self._to_float(mat[i][j])
            except Exception:
                return 0.0

    def _pin_list(self, pins_pos):
        if pins_pos is None:
            return []
        pins = []
        try:
            count = len(pins_pos)
        except Exception:
            return []
        for i in range(count):
            try:
                px = self._to_float(pins_pos[i][0])
                py = self._to_float(pins_pos[i][1])
                pins.append((px, py))
            except Exception:
                continue
        return pins
