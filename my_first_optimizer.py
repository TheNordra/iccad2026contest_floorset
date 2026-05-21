#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied
  - NOT HANDLED: Fixed, preplaced, MIB, cluster, boundary constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i]

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import random
import sys
import json
import os
from pathlib import Path
from typing import List, Tuple
import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026contest.iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree Simulated Annealing baseline.
    
    REPLACE THIS CLASS WITH YOUR ALGORITHM.
    Keep the solve() signature the same.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 20
        self.case_counter = 0

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):
        import math
        import os
        import json

        results = [None] * block_count
        block_types = [0] * block_count
        MAX_WIDTH = 300
        
        # 1. 預置固定塊
        for i in range(block_count):
            tx, ty, tw, th = map(float, target_positions[i])
            if tx != -1 and ty != -1:
                results[i] = [tx, ty, tw, th]
                block_types[i] = 0
            else:
                block_types[i] = 1 if tw != -1 else 2

        # 碰撞檢查函數
        def is_overlap(x, y, w, h, current_results):
            for res in current_results:
                if res is None: continue
                rx, ry, rw, rh = res
                if not (x + w <= rx or x >= rx + rw or y + h <= ry or y >= ry + rh):
                    return True
            return False

        # --- 新增：引力中心計算 (Force-Directed Target) ---
        def calculate_gravity_center(idx, p2b, pins, b2b, current_results):
            sum_x, sum_y, total_weight = 0.0, 0.0, 0.0
            
            # 計算來自 Pin 的引力
            if p2b is not None:
                for edge in p2b:
                    pin_idx, blk_idx = int(edge[0]), int(edge[1])
                    weight = float(edge[2]) if len(edge) > 2 else 1.0
                    if blk_idx == idx:
                        px, py = float(pins[pin_idx][0]), float(pins[pin_idx][1])
                        sum_x += px * weight
                        sum_y += py * weight
                        total_weight += weight
                        
            # 計算來自已放置方塊的引力
            if b2b is not None:
                for edge in b2b:
                    b1, b2 = int(edge[0]), int(edge[1])
                    weight = float(edge[2]) if len(edge) > 2 else 1.0
                    target_b = b2 if b1 == idx else (b1 if b2 == idx else -1)
                    
                    if target_b != -1 and current_results[target_b] is not None:
                        tx, ty, tw, th = current_results[target_b]
                        sum_x += (tx + tw / 2.0) * weight
                        sum_y += (ty + th / 2.0) * weight
                        total_weight += weight
                        
            if total_weight > 0:
                return sum_x / total_weight, sum_y / total_weight
            return -1, -1 # 沒有連線，或者連線對象還沒放置
        # ------------------------------------------------

        # 2. 排序：大面積優先放置
        free_indices = [i for i in range(block_count) if results[i] is None]
        free_indices.sort(key=lambda x: (block_types[x] == 1, area_targets[x]), reverse=True)

        # 3. 引力感知放置 (Gravity-Aware Placement)
        for idx in free_indices:
            area = float(area_targets[idx])
            
            if block_types[idx] == 1:
                w, h = float(target_positions[idx, 2]), float(target_positions[idx, 3])
            else:
                w = math.sqrt(area * 1.5) 
                h = area / w

            # 計算該方塊的理想中心點
            target_x, target_y = calculate_gravity_center(idx, p2b_connectivity, pins_pos, b2b_connectivity, results)
            
            best_pos = None
            best_score = float('inf')
            found_y_baseline = -1

            # 開始掃描空位
            for y_ptr in range(0, 1000, 2):
                # 效能鎖：如果已經找到空位，往上找 30 單位都沒有更好的引力點，就提早結束 (保證 1 秒內)
                if found_y_baseline != -1 and y_ptr > found_y_baseline + 30:
                    break
                    
                for x_ptr in range(0, MAX_WIDTH - int(w) + 1, 4):
                    if not is_overlap(x_ptr, y_ptr, w, h, results):
                        
                        # Soft Macro 形狀適應
                        test_w, test_h = w, h
                        if block_types[idx] == 2:
                            if x_ptr + w * 1.2 <= MAX_WIDTH and not is_overlap(x_ptr, y_ptr, w * 1.2, area / (w * 1.2), results):
                                test_w = w * 1.2
                                test_h = area / test_w
                                
                        cand_cx = x_ptr + test_w / 2.0
                        cand_cy = y_ptr + test_h / 2.0
                        
                        # --- 核心評分邏輯 ---
                        if target_x != -1:
                            # 綜合評估：與目標的曼哈頓距離 (滿足你的向量引力) + 輕度向下重力 (避免亂飄導致面積暴增)
                            distance_penalty = abs(cand_cx - target_x) + abs(cand_cy - target_y)
                            total_score = distance_penalty * 1.0 + y_ptr * 0.2
                        else:
                            # 沒有連線目標，純粹向下堆疊填滿空白
                            total_score = y_ptr * 1.0 
                            
                        if total_score < best_score:
                            best_score = total_score
                            best_pos = [float(x_ptr), float(y_ptr), float(test_w), float(test_h)]
                            if found_y_baseline == -1:
                                found_y_baseline = y_ptr

            if best_pos:
                results[idx] = best_pos
            else:
                results[idx] = [0.0, 800.0, w, h] # 備用方案

        # 4. 輸出視覺化資料
        viz_data = {"test_id": self.case_counter, "block_count": block_count, "positions": results, "block_types": block_types}
        os.makedirs("viz_results", exist_ok=True)
        with open(f"viz_results/case_{self.case_counter}.json", "w") as f:
            json.dump(viz_data, f)
        self.case_counter += 1

        return [tuple(r) for r in results]
        
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
