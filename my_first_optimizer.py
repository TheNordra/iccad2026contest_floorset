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
    
    # def solve(
    #     self,
    #     block_count: int,
    #     area_targets: torch.Tensor,
    #     b2b_connectivity: torch.Tensor,
    #     p2b_connectivity: torch.Tensor,
    #     pins_pos: torch.Tensor,
    #     constraints: torch.Tensor,
    #     target_positions: torch.Tensor = None
    # ) -> List[Tuple[float, float, float, float]]:
    #     """
    #     B*-tree SA optimization.
        
    #     REPLACE THIS METHOD with your algorithm.
    #     Must return List[(x, y, w, h)] with exactly block_count entries.
    #     """
    #     # Initialize dimensions: use target dimensions for fixed/preplaced
    #     # blocks, otherwise start with a square matching the area target.
    #     widths, heights = [], []
    #     for i in range(block_count):
    #         if (target_positions is not None and
    #                 target_positions[i, 2] != -1 and target_positions[i, 3] != -1):
    #             w = float(target_positions[i, 2])
    #             h = float(target_positions[i, 3])
    #         else:
    #             area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
    #             w = h = math.sqrt(area)
    #         widths.append(w)
    #         heights.append(h)
        
    #     # Build B*-tree
    #     tree = BStarTree(block_count, widths, heights)
    #     current_positions = tree.pack()
    #     current_cost = self._cost(current_positions, b2b_connectivity, p2b_connectivity, pins_pos)
        
    #     best_tree = tree.copy()
    #     best_positions = current_positions
    #     best_cost = current_cost
        
    #     # Simulated Annealing
    #     temp = self.initial_temp
    #     while temp > self.final_temp:
    #         for _ in range(self.moves_per_temp):
    #             old_tree = tree.copy()
                
    #             # Random move (only rotate and delete-insert to preserve area)
    #             move = random.randint(0, 1)
    #             if move == 0:
    #                 # Rotate: swap w/h (preserves area w*h)
    #                 tree.move_rotate(random.randint(0, block_count - 1))
    #             else:
    #                 # Delete-insert: move block to new tree position (preserves area)
    #                 tree.move_delete_insert(random.randint(0, block_count - 1))
                
    #             new_positions = tree.pack()
    #             new_cost = self._cost(new_positions, b2b_connectivity, p2b_connectivity, pins_pos)
                
    #             # Accept/reject
    #             delta = new_cost - current_cost
    #             if delta < 0 or random.random() < math.exp(-delta / temp):
    #                 current_positions = new_positions
    #                 current_cost = new_cost
    #                 if current_cost < best_cost:
    #                     best_cost = current_cost
    #                     best_positions = new_positions
    #                     best_tree = tree.copy()
    #             else:
    #                 tree = old_tree
            
    #         temp *= self.cooling_rate
        
    #     return best_positions


    # version 2
    # def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):

    #     results = [None] * block_count
    #     block_types = [0] * block_count
        
    #     # 1. 識別固定方塊與初始邊界
    #     fixed_max_x = 0.0
    #     for i in range(block_count):
    #         tx, ty = float(target_positions[i, 0]), float(target_positions[i, 1])
    #         tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            
    #         if tx != -1 and ty != -1 and tw != -1 and th != -1:
    #             results[i] = (tx, ty, tw, th)
    #             block_types[i] = 0 # Fixed
    #             fixed_max_x = max(fixed_max_x, tx + tw)
    #         elif tw != -1 and th != -1:
    #             block_types[i] = 1 # Hard
    #         else:
    #             block_types[i] = 2 # Soft

    #     # 2. 設定 Bin Packing 參數
    #     # 為了壓縮面積，我們縮減 MAX_WIDTH 並移除間距 (Gap = 0)
    #     MAX_WIDTH = 250.0  
    #     curr_x = fixed_max_x # 起點緊貼固定塊
    #     curr_y = 0.0
    #     row_max_h = 0.0
    #     start_x_for_row = curr_x
    #     GAP = 0.0 # 實現共邊，移除浪費空間

    #     # 3. 排序優化 (關鍵步驟)：先放高的方塊，行高會更整齊
    #     # 建立一個索引清單，排除固定塊後按高度降序排序
    #     free_blocks = []
    #     for i in range(block_count):
    #         if results[i] is None:
    #             if block_types[i] == 1: # Hard
    #                 h = float(target_positions[i, 3])
    #             else: # Soft
    #                 h = math.sqrt(float(area_targets[i]))
    #             free_blocks.append((i, h))
        
    #     # 按高度從高到低排序 (Decreasing Height)
    #     free_blocks.sort(key=lambda x: x[1], reverse=True)

    #     # 4. 開始填充
    #     for i, _ in free_blocks:
    #         # 取得寬高
    #         if block_types[i] == 1: # Hard
    #             w, h = float(target_positions[i, 2]), float(target_positions[i, 3])
    #         else: # Soft
    #             area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
    #             w = h = math.sqrt(area)
            
    #         # 換行檢查
    #         if curr_x + w > start_x_for_row + MAX_WIDTH:
    #             curr_x = start_x_for_row
    #             curr_y += row_max_h # 緊貼上一行頂部換行
    #             row_max_h = 0.0
            
    #         # 執行擺放 (座標完全共邊)
    #         results[i] = (curr_x, curr_y, w, h)
            
    #         # 更新狀態
    #         row_max_h = max(row_max_h, h)
    #         curr_x += w # 下一個方塊起點就是目前方塊的終點
            
    #     # 5. 產出視覺化 JSON
    #     current_id = self.case_counter
    #     self.case_counter += 1
    #     viz_data = {"test_id": current_id, "block_count": block_count, "positions": results, "block_types": block_types}
    #     os.makedirs("viz_results", exist_ok=True)
    #     with open(f"viz_results/case_{current_id}.json", "w") as f:
    #         json.dump(viz_data, f)

    #     return [tuple(r) for r in results]


    # version 3
    # def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):

    #     results = [None] * block_count
    #     block_types = [0] * block_count
        
    #     # 1. 識別固定方塊與初始邊界
    #     fixed_max_x = 0.0
    #     for i in range(block_count):
    #         tx, ty = float(target_positions[i, 0]), float(target_positions[i, 1])
    #         tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            
    #         if tx != -1 and ty != -1 and tw != -1 and th != -1:
    #             results[i] = (tx, ty, tw, th)
    #             block_types[i] = 0 # Fixed
    #             fixed_max_x = max(fixed_max_x, tx + tw)
    #         elif tw != -1 and th != -1:
    #             block_types[i] = 1 # Hard
    #         else:
    #             block_types[i] = 2 # Soft

    #     # 2. 設定 Bin Packing 參數
    #     # 為了壓縮面積，我們縮減 MAX_WIDTH 並移除間距 (Gap = 0)
    #     MAX_WIDTH = 250.0  
    #     curr_x = fixed_max_x # 起點緊貼固定塊
    #     curr_y = 0.0
    #     row_max_h = 0.0
    #     start_x_for_row = curr_x
    #     GAP = 0.0 # 實現共邊，移除浪費空間

    #     # 3. 排序優化 (關鍵步驟)：先放高的方塊，行高會更整齊
    #     # 建立一個索引清單，排除固定塊後按高度降序排序
    #     free_blocks = []
    #     for i in range(block_count):
    #         if results[i] is None:
    #             if block_types[i] == 1: # Hard
    #                 h = float(target_positions[i, 3])
    #             else: # Soft
    #                 h = math.sqrt(float(area_targets[i]))
    #             free_blocks.append((i, h))
        
    #     # 按高度從高到低排序 (Decreasing Height)
    #     free_blocks.sort(key=lambda x: x[1], reverse=True)

    #     # 4. 開始填充
    #     for i, _ in free_blocks:
    #         # 取得寬高
    #         if block_types[i] == 1: # Hard
    #             w, h = float(target_positions[i, 2]), float(target_positions[i, 3])
    #         else: # Soft
    #             area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
    #             w = h = math.sqrt(area)
            
    #         # 換行檢查
    #         if curr_x + w > start_x_for_row + MAX_WIDTH:
    #             curr_x = start_x_for_row
    #             curr_y += row_max_h # 緊貼上一行頂部換行
    #             row_max_h = 0.0
            
    #         # 執行擺放 (座標完全共邊)
    #         results[i] = (curr_x, curr_y, w, h)
            
    #         # 更新狀態
    #         row_max_h = max(row_max_h, h)
    #         curr_x += w # 下一個方塊起點就是目前方塊的終點
            
    #     # 5. 產出視覺化 JSON
    #     current_id = self.case_counter
    #     self.case_counter += 1
    #     viz_data = {"test_id": current_id, "block_count": block_count, "positions": results, "block_types": block_types}
    #     os.makedirs("viz_results", exist_ok=True)
    #     with open(f"viz_results/case_{current_id}.json", "w") as f:
    #         json.dump(viz_data, f)

    #     return [tuple(r) for r in results]


    # version 4
    # def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):
    #     import math
    #     import os
    #     import json

    #     results = [None] * block_count
    #     block_types = [0] * block_count
        
    #     # 1. 識別固定方塊與初始邊界
    #     fixed_blocks = []
    #     for i in range(block_count):
    #         tx, ty = float(target_positions[i, 0]), float(target_positions[i, 1])
    #         tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            
    #         if tx != -1 and ty != -1 and tw != -1 and th != -1:
    #             results[i] = (tx, ty, tw, th)
    #             block_types[i] = 0 # Fixed
    #             fixed_blocks.append((tx, ty, tw, th))
    #         elif tw != -1 and th != -1:
    #             block_types[i] = 1 # Hard
    #         else:
    #             block_types[i] = 2 # Soft

    #     # 2. 排序優化：按高度降序，確保行高由最高的方塊決定
    #     free_blocks = []
    #     for i in range(block_count):
    #         if results[i] is None:
    #             h = float(target_positions[i, 3]) if block_types[i] == 1 else math.sqrt(float(area_targets[i]))
    #             free_blocks.append((i, h))
    #     free_blocks.sort(key=lambda x: x[1], reverse=True)

    #     # 3. 填充邏輯：優先填補固定方塊上方的空間，並調整 Soft Macro 比例
    #     MAX_WIDTH = 300.0
    #     curr_x = 0.0
    #     curr_y = 0.0
    #     row_max_h = 0.0
        
    #     # 處理 17 號等固定方塊上方的填補 (假設從 X=0 開始掃描)
    #     # 這裡簡化為：將自由方塊從 X=0 開始排，碰到固定方塊就跳過重疊區域
    #     for idx, _ in free_blocks:
    #         if results[idx] is not None: continue
            
    #         # 取得原始面積與寬高
    #         if block_types[idx] == 1: # Hard
    #             w, h = float(target_positions[idx, 2]), float(target_positions[idx, 3])
    #         else: # Soft: 初始嘗試高度對齊
    #             area = float(area_targets[idx]) if area_targets[idx] > 0 else 1.0
    #             # 如果是該行第一個方塊，設為正方形定出行高；否則對齊行高
    #             if row_max_h == 0:
    #                 w = h = math.sqrt(area)
    #                 row_max_h = h
    #             else:
    #                 h = row_max_h # 強制對齊行高
    #                 w = area / h  # 調整寬度以維持面積不變

    #         # 換行檢查
    #         if curr_x + w > MAX_WIDTH:
    #             curr_x = 0.0
    #             curr_y += row_max_h
    #             row_max_h = 0.0
    #             # 換行後重新計算第一個方塊的寬高
    #             if block_types[idx] == 2:
    #                 w = h = math.sqrt(area)
    #                 row_max_h = h
    #             else:
    #                 row_max_h = h

    #         # 簡單碰撞檢查：如果與固定方塊重疊，則 X 移到固定方塊右側
    #         for (fx, fy, fw, fh) in fixed_blocks:
    #             # 檢查 Y 軸是否有交集
    #             if not (curr_y + h <= fy or curr_y >= fy + fh):
    #                 # 檢查 X 軸是否重疊
    #                 if not (curr_x + w <= fx or curr_x >= fx + fw):
    #                     curr_x = fx + fw # 移到右側

    #         results[idx] = (curr_x, curr_y, w, h)
    #         curr_x += w

    #     # 4. 產出視覺化與回傳
    #     current_id = self.case_counter
    #     self.case_counter += 1
    #     viz_data = {"test_id": current_id, "block_count": block_count, "positions": results, "block_types": block_types}
    #     os.makedirs("viz_results", exist_ok=True)
    #     with open(f"viz_results/case_{current_id}.json", "w") as f:
    #         json.dump(viz_data, f)

    #     return [tuple(r) for r in results]


    # version 5
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):
        import math
        import os
        import json

        results = [None] * block_count
        block_types = [0] * block_count
        
        # 1. 初始資料整理
        for i in range(block_count):
            tx, ty, tw, th = map(float, target_positions[i])
            if tx != -1 and ty != -1:
                results[i] = [tx, ty, tw, th]
                block_types[i] = 0
            else:
                block_types[i] = 1 if tw != -1 else 2

        free_indices = [i for i in range(block_count) if results[i] is None]
        # 按面積降序，大塊先放
        free_indices.sort(key=lambda x: area_targets[x], reverse=True)

        # 2. 定義一個精確的碰撞檢查函數
        def is_overlap(x, y, w, h, current_results):
            for res in current_results:
                if res is None: continue
                rx, ry, rw, rh = res
                # 檢查矩形交集
                if not (x + w <= rx or x >= rx + rw or y + h <= ry or y >= ry + rh):
                    return True
            return False

        # 3. 填充循環
        MAX_WIDTH = 300.0
        STEP = 2.0 # 步進值，越小越精確但越慢

        for idx in free_indices:
            area = float(area_targets[idx])
            # 初始嘗試：維持一個合理的 Aspect Ratio (例如 0.5 ~ 2.0)
            best_h = math.sqrt(area)
            best_w = area / best_h
            
            found = False
            # 由下而上 (y)，由左向右 (x) 搜尋
            for y_top in range(0, 500, int(STEP)):
                for x_left in range(0, int(MAX_WIDTH), int(STEP)):
                    # 測試 Soft Macro 是否可以透過變形塞入目前的 y 高度
                    # 我們嘗試三種比例：瘦長、正方、扁平
                    for ratio in [1.5, 1.0, 0.6]:
                        test_h = math.sqrt(area * ratio)
                        test_w = area / test_h
                        
                        if x_left + test_w <= MAX_WIDTH:
                            if not is_overlap(x_left, y_top, test_w, test_h, results):
                                results[idx] = [float(x_left), float(y_top), test_w, test_h]
                                found = True
                                break
                    if found: break
                if found: break

        # 4. 輸出
        viz_data = {"test_id": self.case_counter, "block_count": block_count, "positions": results, "block_types": block_types}
        self.case_counter += 1
        os.makedirs("viz_results", exist_ok=True)
        with open(f"viz_results/case_{viz_data['test_id']}.json", "w") as f:
            json.dump(viz_data, f)
            
        return [tuple(r) for r in results]
    
    
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
