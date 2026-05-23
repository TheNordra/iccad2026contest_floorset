#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Training Data Example

Shows how to train a neural network using the DIFFERENTIABLE contest cost function.
Run: python iccad2026contest/training_example.py
<<<<<<< HEAD
=======

The loss approximates the contest evaluation formula:
  Cost ≈ (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)

Note: RuntimeFactor (× max(0.7, R^γ)) is omitted — it is not available during
training. The full contest cost is evaluated server-side using per-test-case
median runtimes across all submissions. V_soft here is a differentiable proxy
(overlap area + area-tolerance excess), not the exact grouping/MIB/boundary
violation counts used in final scoring.

But implemented with differentiable operations for .backward()
>>>>>>> origin/main
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)

# =========================================================================
# 定義圖神經網路 (Graph Neural Network)
# =========================================================================
class FloorplanNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 特徵融合層：接收 [面積, 引力Pin_X, 引力Pin_Y]
        self.input_layer = nn.Linear(3, hidden_dim)
        
        # 圖卷積層 (Graph Convolution)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出層：預測 (x, y, w, h)
        self.out_layer = nn.Linear(hidden_dim, 4)

    def forward(self, area_target, b2b_conn, p2b_conn, pins_pos, block_count):
        device = area_target.device
        
        # 1. 建立初始節點特徵 [面積, 引力X, 引力Y]
        features = torch.zeros((block_count, 3), device=device)
        features[:, 0] = area_target[:block_count]
        
        pin_counts = torch.zeros(block_count, device=device)
        if p2b_conn.dim() == 2 and len(p2b_conn) > 0:
            for edge in p2b_conn:
                if edge[0] == -1: continue 
                pin_idx, blk_idx = int(edge[0]), int(edge[1])
                if blk_idx < block_count:
                    features[blk_idx, 1] += pins_pos[pin_idx, 0]
                    features[blk_idx, 2] += pins_pos[pin_idx, 1]
                    pin_counts[blk_idx] += 1
        
        mask = pin_counts > 0
        features[mask, 1] /= pin_counts[mask]
        features[mask, 2] /= pin_counts[mask]

        # 2. 轉換為隱藏層向量
        x = torch.relu(self.input_layer(features))
        
        # 3. 建立相鄰矩陣
        adj = torch.eye(block_count, device=device)
        if b2b_conn.dim() == 2 and len(b2b_conn) > 0:
            for edge in b2b_conn:
                if edge[0] == -1: continue
                u, v = int(edge[0]), int(edge[1])
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                if u < block_count and v < block_count:
                    adj[u, v] += weight
                    adj[v, u] += weight
                    
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / (row_sum + 1e-8)

        # 4. 圖卷積
        x = torch.matmul(adj, x)
        x = torch.relu(self.gcn1(x))
        x = torch.matmul(adj, x)
        x = torch.relu(self.gcn2(x))

        # 5. 輸出層
        out = self.out_layer(x)
        xy = out[:, :2]
        wh = torch.nn.functional.softplus(out[:, 2:]) # 確保寬高為正數
        
        return torch.cat([xy, wh], dim=1)


def main():
    print("="*70)
    print("ICCAD 2026 FloorSet Challenge - GNN Training")
    print("="*70)
    
    # 1. 載入資料
    print("\nLoading training data...")
    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=100,  # 建議先設定 100 筆測試，確定能跑再改成 None (全部訓練)
        shuffle=True
    )
    print(f"Loaded {len(dataloader)} samples\n")
    
    # 2. 初始化神經網路與優化器 (移到迴圈外，只初始化一次)
    # 偵測是否有 GPU 可以加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = FloorplanNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train() # 進入訓練模式

    # 3. 開始訓練迴圈
    for batch_idx, batch in enumerate(dataloader):
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
        
        # 降維並移至 GPU/CPU
        area_target = area_target.squeeze(0).to(device)
        b2b_conn = b2b_conn.squeeze(0).to(device)
        p2b_conn = p2b_conn.squeeze(0).to(device)
        pins_pos = pins_pos.squeeze(0).to(device)
        metrics = metrics.squeeze(0).to(device)
        
        block_count = int((area_target != -1).sum().item())
        print(f"Epoch/Sample {batch_idx}: {block_count} blocks")
        
        # --- 神經網路前向傳播 (預測) ---
        optimizer.zero_grad()
        positions = model(area_target, b2b_conn, p2b_conn, pins_pos, block_count)
        
        # --- 計算官方 Differentiable Loss ---
        loss = compute_training_loss_differentiable(
            positions,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target[:block_count],
            metrics
        )
        
        print(f"  -> Loss: {loss.item():.4f}")
        
        # --- 反向傳播與參數更新 ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print("="*70)
    print("Training loop finished!")
    
    # 儲存訓練好的模型權重
    torch.save(model.state_dict(), "floorplan_gnn.pth")
    print("Model saved to floorplan_gnn.pth")
    print("="*70)

if __name__ == '__main__':
    main()