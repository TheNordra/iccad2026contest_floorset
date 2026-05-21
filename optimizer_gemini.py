#!/usr/bin/env python3
import os
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)

class MyOptimizer(FloorplanOptimizer):
    """
    ICCAD 2026 FloorSet Challenge - V6 鐵壁版 B*Tree SA
    修復：MIB 連動導致 Fixed/Preplaced 區塊面積被異常覆蓋的 Bug
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.case_counter = 0
        
        # 退火參數：穩定收斂配置
        self.initial_temp = 5000.0
        self.final_temp = 0.5
        self.cooling_rate = 0.90
        self.moves_per_temp = 50

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        
        # --- 1. 資料預處理 ---
        b2b_list = []
        if b2b_connectivity is not None and len(b2b_connectivity) > 0:
            b2b_list = [(int(edge[0]), int(edge[1]), float(edge[2])) for edge in b2b_connectivity]
            
        p2b_list = []
        if p2b_connectivity is not None and len(p2b_connectivity) > 0:
            p2b_list = [(int(edge[0]), int(edge[1]), float(edge[2])) for edge in p2b_connectivity]
            
        pins_list = []
        if pins_pos is not None and len(pins_pos) > 0:
            pins_list = [(float(p[0]), float(p[1])) for p in pins_pos]

        base_areas = [float(a) if a > 0 else 1.0 for a in area_targets]

        # --- 2. 嚴格解析 Constraints 與 形狀鎖定 ---
        is_fixed_shape = [False] * block_count # 👑 核心防護：記錄是否允許變形/旋轉
        is_preplaced = [False] * block_count
        block_types = [2] * block_count
        
        widths = [0.0] * block_count
        heights = [0.0] * block_count
        preplaced_coords = {}
        mib_groups = {}
        
        for i in range(block_count):
            c_fixed = int(constraints[i, 0].item()) if constraints is not None else 0
            c_preplaced = int(constraints[i, 1].item()) if constraints is not None else 0
            c_mib = int(constraints[i, 2].item()) if constraints is not None else -1
            
            if c_mib != -1:
                mib_groups.setdefault(c_mib, []).append(i)
                
            target_x, target_y, target_w, target_h = -1.0, -1.0, -1.0, -1.0
            if target_positions is not None:
                target_x, target_y = float(target_positions[i, 0]), float(target_positions[i, 1])
                target_w, target_h = float(target_positions[i, 2]), float(target_positions[i, 3])

            if c_preplaced == 1 and target_w != -1:
                is_preplaced[i] = True
                is_fixed_shape[i] = True  # Preplaced 絕對不允許變形
                block_types[i] = 0
                widths[i], heights[i] = target_w, target_h
                preplaced_coords[i] = (target_x, target_y)
            elif c_fixed == 1 and target_w != -1:
                is_fixed_shape[i] = True  # Fixed 絕對不允許變形
                block_types[i] = 1
                widths[i], heights[i] = target_w, target_h
            else:
                block_types[i] = 2
                if target_w != -1 and target_h != -1:
                    widths[i], heights[i] = target_w, target_h
                else:
                    widths[i] = math.sqrt(base_areas[i])
                    heights[i] = base_areas[i] / widths[i]

        # 👑 MIB 病毒式上鎖：若群組內有 Fixed/Preplaced，全部強制同化並上鎖
        for mib_id, blocks in mib_groups.items():
            if not blocks: continue
            
            fixed_idx = -1
            for b in blocks:
                if is_fixed_shape[b]:
                    fixed_idx = b
                    break
                    
            if fixed_idx != -1:
                base_w, base_h = widths[fixed_idx], heights[fixed_idx]
                for b in blocks:
                    widths[b], heights[b] = base_w, base_h
                    is_fixed_shape[b] = True # 一人 Fixed，全家 Fixed
            else:
                base_w, base_h = widths[blocks[0]], heights[blocks[0]]
                for b in blocks:
                    widths[b], heights[b] = base_w, base_h

        # 建立 B*Tree 拓樸 (排除 Preplaced)
        tree_blocks = [i for i in range(block_count) if not is_preplaced[i]]
        n_tree = len(tree_blocks)
        left = [-1] * n_tree
        right = [-1] * n_tree
        
        if n_tree > 0:
            for i in range(1, n_tree):
                parent = random.randint(0, i - 1)
                if random.random() < 0.5:
                    left[i] = left[parent]
                    left[parent] = i
                else:
                    right[i] = right[parent]
                    right[parent] = i
                    
        # --- 3. 天際線打包引擎 (Skyline Packer) ---
        def pack(current_widths, current_heights, current_tree):
            pos = [[-1.0, -1.0, 0.0, 0.0] for _ in range(block_count)]
            contour = [[0.0, 1e9, 0.0]]
            
            def get_contour_y(x1, x2):
                max_y = 0.0
                for s_x1, s_x2, s_y in contour:
                    if max(x1, s_x1) < min(x2, s_x2) - 1e-6:
                        max_y = max(max_y, s_y)
                return max_y

            def update_contour(x1, x2, y_top):
                nonlocal contour
                new_contour = []
                for s_x1, s_x2, s_y in contour:
                    if s_x2 <= x1 or s_x1 >= x2:
                        new_contour.append([s_x1, s_x2, s_y])
                    else:
                        if s_x1 < x1: new_contour.append([s_x1, x1, s_y])
                        if s_x2 > x2: new_contour.append([x2, s_x2, s_y])
                new_contour.append([x1, x2, y_top])
                new_contour.sort(key=lambda s: s[0])
                
                merged = []
                for s in new_contour:
                    if merged and abs(merged[-1][2] - s[2]) < 1e-6 and merged[-1][1] >= s[0] - 1e-6:
                        merged[-1][1] = max(merged[-1][1], s[1])
                    else:
                        merged.append(s)
                contour = merged

            # 注入 Preplaced 障礙物
            for i in range(block_count):
                if is_preplaced[i]:
                    px, py = preplaced_coords[i]
                    pw, ph = current_widths[i], current_heights[i]
                    pos[i] = [px, py, pw, ph]
                    update_contour(px, px + pw, py + ph)

            def dfs(node_idx, parent_right_edge):
                if node_idx == -1: return
                b_idx = current_tree[node_idx]
                w, h = current_widths[b_idx], current_heights[b_idx]
                
                x = 0.0 if node_idx == 0 else parent_right_edge
                y = get_contour_y(x, x + w)
                
                pos[b_idx] = [x, y, w, h]
                update_contour(x, x + w, y + h)
                
                dfs(left[node_idx], x + w)
                dfs(right[node_idx], x)
            
            if n_tree > 0:
                dfs(0, 0.0)
                
            return pos

        # --- 4. 成本函數 ---
        def evaluate_cost(pos):
            hpwl = 0.0
            for i_idx, j_idx, weight in b2b_list:
                cx1 = pos[i_idx][0] + pos[i_idx][2] * 0.5
                cy1 = pos[i_idx][1] + pos[i_idx][3] * 0.5
                cx2 = pos[j_idx][0] + pos[j_idx][2] * 0.5
                cy2 = pos[j_idx][1] + pos[j_idx][3] * 0.5
                hpwl += weight * (abs(cx1 - cx2) + abs(cy1 - cy2))
                    
            for p_idx, b_idx, weight in p2b_list:
                px, py = pins_list[p_idx]
                cx = pos[b_idx][0] + pos[b_idx][2] * 0.5
                cy = pos[b_idx][1] + pos[b_idx][3] * 0.5
                hpwl += weight * (abs(px - cx) + abs(py - cy))
            
            min_x = min(p[0] for p in pos)
            min_y = min(p[1] for p in pos)
            max_x = max(p[0] + p[2] for p in pos)
            max_y = max(p[1] + p[3] for p in pos)
            area = (max_x - min_x) * (max_y - min_y)
            
            return hpwl + area * 0.5

        current_tree = tree_blocks.copy()
        current_w, current_h = widths.copy(), heights.copy()
        
        current_pos = pack(current_w, current_h, current_tree)
        current_cost = evaluate_cost(current_pos)
        
        best_pos = [list(p) for p in current_pos]
        best_cost = current_cost
        
        # --- 5. SA 退火迴圈 ---
        temp = self.initial_temp
        while temp > self.final_temp:
            for _ in range(self.moves_per_temp):
                if n_tree == 0: break
                
                action = random.choice(["swap", "reshape", "rotate"])
                
                old_tree = current_tree.copy()
                old_w, old_h = current_w.copy(), current_h.copy()
                
                idx = random.randint(0, n_tree - 1)
                b_idx = current_tree[idx]
                
                if action == "swap" and n_tree >= 2:
                    idx1, idx2 = random.sample(range(n_tree), 2)
                    current_tree[idx1], current_tree[idx2] = current_tree[idx2], current_tree[idx1]
                    
                elif action == "rotate":
                    if not is_fixed_shape[b_idx]: # 👑 安全防護：絕對不轉 Fixed
                        current_w[b_idx], current_h[b_idx] = current_h[b_idx], current_w[b_idx]
                        for mib_id, blocks in mib_groups.items():
                            if b_idx in blocks:
                                for sibling in blocks:
                                    current_w[sibling], current_h[sibling] = current_w[b_idx], current_h[b_idx]
                                    
                elif action == "reshape":
                    if not is_fixed_shape[b_idx]: # 👑 安全防護：絕對不變形 Fixed
                        target_area = base_areas[b_idx]
                        ratio = random.uniform(0.5, 2.0)
                        
                        new_w = math.sqrt(target_area * ratio)
                        new_h = target_area / new_w
                        
                        current_w[b_idx], current_h[b_idx] = new_w, new_h
                        
                        for mib_id, blocks in mib_groups.items():
                            if b_idx in blocks:
                                for sibling in blocks:
                                    current_w[sibling] = new_w
                                    current_h[sibling] = new_h

                new_pos = pack(current_w, current_h, current_tree)
                new_cost = evaluate_cost(new_pos)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos = [list(p) for p in new_pos]
                else:
                    current_tree = old_tree
                    current_w, current_h = old_w, old_h

            temp *= self.cooling_rate

        # --- 6. 輸出 JSON ---
        results = [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in best_pos]

        viz_data = {
            "test_id": self.case_counter, 
            "block_count": block_count, 
            "positions": results, 
            "block_types": block_types
        }
        os.makedirs("viz_results", exist_ok=True)
        with open(f"viz_results/case_{self.case_counter}.json", "w") as f:
            json.dump(viz_data, f)
        self.case_counter += 1

        return results