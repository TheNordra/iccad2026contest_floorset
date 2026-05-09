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

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ):
        # version 1: error on fixed info
        # results = [None] * block_count
        # free_block_indices = []

        # # --- 1. 先處理固定方塊 ---
        # for i in range(block_count):
        #     is_fixed = constraints[i, 0] == 1
        #     is_preplaced = constraints[i, 1] == 1
            
        #     if is_fixed or is_preplaced:
        #         # 取得固定座標 (x, y, w, h)
        #         x = float(target_positions[i, 0])
        #         y = float(target_positions[i, 1])
        #         w = float(target_positions[i, 2])
        #         h = float(target_positions[i, 3])

        #         if x < 0 or y < 0 or w < 0 or h < 0:
        #             print("#438 error: wrong result")

        #         results[i] = (x, y, w, h)
        #     else:
        #         # 這些是我們待會要自己放的
        #         free_block_indices.append(i)

        # # --- 2. 處理自由方塊 (簡易堆疊法) ---
        # # 為了避免重疊到固定方塊，我們可以先找一個「保險區塊」
        # # 例如：所有固定方塊的最右邊或最上方
        # start_x = 0.0
        # if any(r is not None for r in results):
        #     # 找到目前所有固定方塊的最右邊界，從那裡開始排，避免重疊
        #     start_x = max(r[0] + r[2] for r in results if r is not None)

        # current_x = start_x
        # current_y = 0.0
        # max_h_in_row = 0.0

        # for i in free_block_indices:
        #     area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
        #     w = h = math.sqrt(area)
            
        #     # 這裡可以加入換行邏輯 (假設一個很大的寬度)
        #     if current_x + w > start_x + 2000: # 暫定寬度
        #         current_x = start_x
        #         current_y += max_h_in_row
        #         max_h_in_row = 0.0
            
        #     results[i] = (current_x, current_y, w, h)
        #     current_x += w
        #     max_h_in_row = max(max_h_in_row, h)

        # return results


        # version 2
        results = [None] * block_count
        to_be_placed = [] # 儲存需要我們決定位置的 index

        for i in range(block_count):
            tx, ty = float(target_positions[i, 0]), float(target_positions[i, 1])
            tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])

            # 類型 1: 完全固定 (座標與長寬都有)
            if tx != -1 and ty != -1 and tw != -1 and th != -1:
                results[i] = (tx, ty, tw, th)
            
            # 類型 2: 固定長寬，但沒位置 (Hard Macro)
            elif tw != -1 and th != -1 and tx == -1 and ty == -1:
                results[i] = [None, None, tw, th] # 座標待定
                to_be_placed.append(i)
                
            # 類型 3: 只有面積，長寬位置都沒 (Soft Macro)
            elif tw == -1 and th == -1:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                # 第一版先假設為正方形
                side = math.sqrt(area)
                results[i] = [None, None, side, side] # 座標待定
                to_be_placed.append(i)
            
            else:
                # 預防萬一，印出沒考慮到的特殊情況
                if self.verbose:
                    print(f"Warning: Block {i} has unusual target: {tx, ty, tw, th}")
                # 預設處理
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                side = math.sqrt(area)
                results[i] = [None, None, side, side]
                to_be_placed.append(i)

        # --- 開始簡單擺放 ---
        # 為了避免重疊，我們先找到目前「完全固定」方塊的邊界
        fixed_max_x = 0.0
        for r in results:
            if r is not None and r[0] is not None:
                fixed_max_x = max(fixed_max_x, r[0] + r[2])

        # 從固定方塊的最右邊開始排隊，這是一個絕對安全的「保險策略」
        curr_x, curr_y = fixed_max_x, 0.0
        max_h_in_row = 0.0
        
        for i in to_be_placed:
            w, h = results[i][2], results[i][3]
            
            # 簡單換行邏輯 (先假設一個夠大的 Die 寬度，例如 5000)
            if curr_x + w > fixed_max_x + 5000:
                curr_x = fixed_max_x
                curr_y += max_h_in_row
                max_h_in_row = 0
            
            results[i][0] = curr_x
            results[i][1] = curr_y
            
            curr_x += w
            max_h_in_row = max(max_h_in_row, h)

        # 將 results 轉換回符合要求的 List[Tuple]
        return [tuple(r) for r in results]

    
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
