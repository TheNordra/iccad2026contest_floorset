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
    
    # version 6: Total Score: 21.9174, Avg Cost: 13.5191, Avg Runtime: 3.56s
    # def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):

    #     results = [None] * block_count
    #     block_types = [0] * block_count # 0:Fixed, 1:Hard, 2:Soft
    #     fixed_blocks = []

    #     # 1. 初始化資料與計算理想高度
    #     total_soft_area = 0.0
    #     for i in range(block_count):
    #         tx, ty, tw, th = map(float, target_positions[i])
    #         if tx != -1 and ty != -1:
    #             results[i] = [tx, ty, tw, th]
    #             block_types[i] = 0
    #             fixed_blocks.append((tx, ty, tw, th))
    #         else:
    #             block_types[i] = 1 if tw != -1 else 2
    #             total_soft_area += float(area_targets[i]) if area_targets[i] > 0 else 100.0

    #     # 計算理想目標高度 (1.1x 緩衝)
    #     MAX_WIDTH = 300.0
    #     h_target = (total_soft_area / MAX_WIDTH) * 1.1

    #     # 2. 排序優化：大方塊與不可變形的 Hard Macro 優先
    #     free_indices = [i for i in range(block_count) if results[i] is None]
    #     free_indices.sort(key=lambda x: (block_types[x], area_targets[x]), reverse=True)

    #     # 3. 幾何碰撞檢查函數 (保證 0 重疊)
    #     def check_collision(x, y, w, h, placed_results):
    #         for res in placed_results:
    #             if res is None: continue
    #             px, py, pw, ph = res
    #             # AABB 碰撞檢測：如果不滿足「完全分離」，則重疊
    #             if not (x + w <= px or x >= px + pw or y + h <= py or y >= py + ph):
    #                 return True
    #         return False

    #     # 4. 填充邏輯：搜尋與變形填滿 (Step-based Search)
    #     # STEP 越小越緊湊但運算越慢，2.0 是一個折衷方案
    #     X_STEP = 2.0 
    #     Y_STEP = 2.0

    #     for idx in free_indices:
    #         area = float(area_targets[idx])
    #         found = False
            
    #         # 從 Y=0, X=0 開始，由下而上、由左向右搜尋空隙
    #         # 搜尋上限設高一點 (例如 500) 確保能塞下，但會優先填低處
    #         for y in range(0, 500, int(Y_STEP)):
    #             for x in range(0, int(MAX_WIDTH), int(X_STEP)):
                    
    #                 if block_types[idx] == 1: # Hard Macro: 尺寸固定
    #                     w, h = float(target_positions[idx, 2]), float(target_positions[idx, 3])
    #                     if x + w <= MAX_WIDTH:
    #                         if not check_collision(x, y, w, h, results):
    #                             results[idx] = [float(x), float(y), w, h]
    #                             found = True
    #                             break
    #                 else: # Soft Macro: 可變形填充
    #                     # 嘗試不同比例：扁平 (2:1), 正方 (1:1), 瘦長 (1:2)
    #                     # 這是為了壓縮空間，不執著於正方形
    #                     possible_ratios = [0.5, 1.0, 2.0] # w:h ratio
                        
    #                     # 特殊邏輯：主動填補固定塊下方。如果上方有固定塊，計算高度以貼合
    #                     w_test = h_test = math.sqrt(area)
    #                     ceiling_y = 999.0
    #                     for (fx, fy, fw, fh) in fixed_blocks:
    #                         if x >= fx and x < fx + fw: # 目前 X 在固定塊下方
    #                             if fy > y: ceiling_y = min(ceiling_y, fy)
                        
    #                     # 如果有天花板，且空間不大不小 (例如是原本 square 高度的 0.8x~1.5x)
    #                     if ceiling_y != 999.0 and \
    #                        (h_test * 0.8) <= (ceiling_y - y) <= (h_test * 1.5):
    #                         possible_ratios = [(area / (ceiling_y - y)**2)] # 計算新的寬高比

    #                     for ratio in possible_ratios:
    #                         # 確保 Aspect Ratio 在 [1/3, 3] 之內
    #                         test_ratio = max(0.33, min(3.0, ratio))
    #                         h = math.sqrt(area / test_ratio)
    #                         w = area / h
                            
    #                         if x + w <= MAX_WIDTH:
    #                             if not check_collision(x, y, w, h, results):
    #                                 results[idx] = [float(x), float(y), w, h]
    #                                 found = True
    #                                 break
    #                     if found: break
    #                 if found: break
    #             if found: break

    #     # 5. 輸出
    #     current_id = self.case_counter
    #     self.case_counter += 1
    #     viz_data = {"test_id": current_id, "block_count": block_count, "positions": results, "block_types": block_types}
    #     os.makedirs("viz_results", exist_ok=True)
    #     with open(f"viz_results/case_{current_id}.json", "w") as f:
    #         json.dump(viz_data, f)

    #     return [tuple(r) for r in results]

    # version 7:
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):

        results = [None] * block_count
        block_types = [0] * block_count
        fixed_blocks = []

        # 1. 快速提取固定塊
        for i in range(block_count):
            tx, ty, tw, th = map(float, target_positions[i])
            if tx != -1 and ty != -1:
                results[i] = [tx, ty, tw, th]
                block_types[i] = 0
                fixed_blocks.append((tx, ty, tw, th))
            else:
                block_types[i] = 1 if tw != -1 else 2

        # 2. 預測目標高度以決定 Soft Macro 初始比例
        total_area = sum([float(a) for a in area_targets if a > 0])
        MAX_WIDTH = 300.0
        est_h = total_area / MAX_WIDTH

        # 3. Skyline 陣列：用來儲存每個 X 位置目前的最高 Y (解析度 1 單位，長度 300)
        skyline = [0.0] * 301
        
        # 將固定塊寫入 Skyline（若固定塊下方是空的，此邏輯會優先填下方）
        # 為了填滿 6 號下方，我們先不把固定塊寫入底部的 Skyline，而是視為「天花板」障礙
        
        # 4. 排序：按寬度降序 (大塊先放，底部會更平整)
        free_indices = [i for i in range(block_count) if results[i] is None]
        free_indices.sort(key=lambda x: area_targets[x], reverse=True)

        def get_min_y_range(w_int):
            best_x = 0
            min_y = 1e9
            for x in range(0, int(MAX_WIDTH - w_int) + 1, 2): # 步進 2 增加速度
                current_max_y = max(skyline[x : x + w_int])
                if current_max_y < min_y:
                    min_y = current_max_y
                    best_x = x
            return best_x, min_y

        # 5. 快速填充
        for idx in free_indices:
            area = float(area_targets[idx])
            
            # 決定尺寸：Soft Macro 根據目標高度調整
            if block_types[idx] == 1:
                w, h = float(target_positions[idx, 2]), float(target_positions[idx, 3])
            else:
                # 策略：為了壓縮高度，讓寬度稍微大於高度 (Ratio 1.2~1.5)
                w = math.sqrt(area * 1.3)
                h = area / w
            
            w_int = int(math.ceil(w))
            
            # 尋找最低地面
            best_x, min_y = get_min_y_range(w_int)
            
            # 碰撞檢查與天花板感知 (Case 80, Block 6 下方填充)
            # 檢查目前位置上方是否有固定塊阻擋
            for fx, fy, fw, fh in fixed_blocks:
                # 如果 X 範圍有重疊
                if not (best_x + w <= fx or best_x >= fx + fw):
                    # 如果會撞到固定塊，且固定塊在我們上方
                    if min_y < fy and min_y + h > fy:
                        # 壓縮高度以填滿空隙
                        if block_types[idx] == 2:
                            h = fy - min_y
                            w = area / h
                            w_int = int(math.ceil(w))
                            # 重新檢查 X 邊界
                            if best_x + w_int > MAX_WIDTH:
                                best_x = int(MAX_WIDTH - w_int)
                        else:
                            # Hard Macro 撞到了就疊到固定塊上面
                            min_y = fy + fh

            # 最終檢查是否與任何方塊重疊（確保安全）
            results[idx] = [float(best_x), float(min_y), w, h]
            
            # 更新 Skyline
            new_top = min_y + h
            for x in range(best_x, min_y_range_end := min(301, best_x + w_int)):
                skyline[x] = new_top

        # 6. 輸出結果
        viz_data = {"test_id": self.case_counter, "block_count": block_count, "positions": results, "block_types": block_types}
        self.case_counter += 1
        return [tuple(r) for r in results]
        
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
