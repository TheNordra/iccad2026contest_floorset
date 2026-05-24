#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Training Data Example (GPU & Bounded GNN Version)

Trains a neural network using the DIFFERENTIABLE contest cost function on GPU.
Locks aspect ratio and coordinates to prevent network cheating.
Run: python iccad2026contest/training_example.py
"""

import sys
from pathlib import Path
import math

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)

# =========================================================================
# 定義升級版圖神經網路 (Bounded Graph Neural Network)
# =========================================================================
class FloorplanNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 特徵融合層：接收 [面積, 引力Pin_X, 引力Pin_Y]
        self.input_layer = nn.Linear(3, hidden_dim)
        
        # 圖卷積層 (Graph Convolution)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出層：預測 (x_norm, y_norm, log_ratio, _unused)
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
        
        # 3. 建立相鄰矩陣 (微調平滑項)
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

        # 5. 輸出層與防呆映射
        out = self.out_layer(x)
        
        # 在 forward 映射座標時，加上一個依 block 編號微調的初始散開值
        # 讓模型是在「已經稍微鋪開」的基礎上，再去調整位置，而不是大家都從 (0,0) 開始擠
        grid_x = (torch.arange(block_count, device=device) % 10) * 20.0
        grid_y = (torch.arange(block_count, device=device) // 10) * 20.0
        xy = torch.sigmoid(out[:, :2]) * 150.0 + torch.stack([grid_x, grid_y], dim=1)
        
        # 【手術二】長寬比鎖定：預測 log(ratio)，轉換後範圍限制在 0.1 ~ 10.0 之間
        log_ratio = torch.tanh(out[:, 2]) * math.log(10.0)
        ratio = torch.exp(log_ratio)
        
        # 根據精確目標面積計算真實的 w 和 h
        areas = area_target[:block_count]
        w = torch.sqrt(areas * ratio)
        h = torch.sqrt(areas / ratio)
        
        # 將 w 和 h 重新維度調整為 (N, 1) 並與 xy 拼接
        wh = torch.stack([w, h], dim=1)
        
        return torch.cat([xy, wh], dim=1)


def main():
    print("="*70)
    print("ICCAD 2026 FloorSet Challenge - GNN Training (GPU)")
    print("="*70)
    
    # 1. 載入資料 (可以試著把 num_samples 放大，例如 1000 筆或更多)
    print("\nLoading training data...")
    dataloader = get_training_dataloader(
        batch_size=4,
        num_samples=500,  
        shuffle=True
    )
    print(f"Loaded {len(dataloader)} samples\n")
    
    # 2. 初始化神經網路與優化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 選擇訓練好的模型
    model = FloorplanNet().to(device)
    if Path("floorplan_gnn.pth").exists():
        model.load_state_dict(torch.load("floorplan_gnn.pth", map_location=device))
        print("Successfully loaded trained weights from floorplan_gnn.pth!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 調低 Learning rate 讓收斂更平穩
    model.train()

    # 3. 開始訓練迴圈
    for batch_idx, batch in enumerate(dataloader):
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
        
        current_batch_size = area_target.size(0)
        optimizer.zero_grad()
        total_loss = 0.0
        
        for b in range(current_batch_size):
            b_area = area_target[b].to(device)
            b_b2b = b2b_conn[b].to(device)
            b_p2b = p2b_conn[b].to(device)
            b_pins = pins_pos[b].to(device)
            b_metrics = metrics[b].to(device)
            
            block_count = int((b_area != -1).sum().item())
            
            positions = model(b_area, b_b2b, b_p2b, b_pins, block_count)
            
            loss = compute_training_loss_differentiable(
                positions, b_b2b, b_p2b, b_pins, b_area[:block_count], b_metrics
            )
            total_loss += loss
            
        # 算出平均 Loss
        total_loss = total_loss / current_batch_size
        print(f"Batch {batch_idx} -> Average Loss: {total_loss.item():.4f}")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ==========================================
        # 【新增防禦】每 20 個 Batch 自動存檔一次
        # ==========================================
        if batch_idx > 0 and batch_idx % 20 == 0:
            torch.save(model.state_dict(), "floorplan_gnn_checkpoint.pth")
            print(f"[Checkpoint] 已自動備份權重至 floorplan_gnn_checkpoint.pth (Batch {batch_idx})")

    # =========================================================================
    # 🏁 訓練完全結束：【立刻存檔】（放在繪圖之前，絕對安全）
    # =========================================================================
    print("\n" + "="*70)
    print("Training loop finished successfully!")
    torch.save(model.state_dict(), "floorplan_gnn.pth")
    print("[Final] 最終精準權重已成功安全存檔至 floorplan_gnn.pth")
    print("="*70)

    # =========================================================================
    # 🎨 【修正版】視覺化成果繪圖測試（放在最底層，就算出錯也不影響存檔）
    # =========================================================================
    print("\nGenerating Visualization Result...")
    model.eval()
    
    with torch.no_grad():
        # 安全抽取第一筆資料
        single_area = area_target[0].to(device)
        single_b2b = b2b_conn[0].to(device)
        single_p2b = p2b_conn[0].to(device)
        single_pins = pins_pos[0].to(device)
        single_metrics = metrics[0].to(device)
        
        block_count = int((single_area != -1).sum().item())
        pred_positions = model(single_area, single_b2b, single_p2b, single_pins, block_count)
        
        single_loss = compute_training_loss_differentiable(
            pred_positions, single_b2b, single_p2b, single_pins, single_area[:block_count], single_metrics
        )
        
        pred_np = pred_positions.cpu().numpy()
        pins_np = single_pins.cpu().numpy()
        
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(block_count):
            x, y, w, h = pred_np[i]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.4
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, str(i), fontsize=6, ha='center', va='center')
            
        if len(pins_np) > 0 and pins_np[0, 0] != -1:
            valid_pins = pins_np[pins_np[:, 0] != -1]
            if len(valid_pins) > 0:
                ax.scatter(valid_pins[:, 0], valid_pins[:, 1], c='red', s=15, marker='x', label='Pins')
            
        ax.set_xlim(-10, 310)
        ax.set_ylim(-10, 310)
        ax.set_aspect('equal')
        plt.title(f"Predicted Floorplan Layout (Single Loss: {single_loss.item():.4f})")
        plt.legend()
        
        output_fig_path = "predicted_floorplan.png"
        plt.savefig(output_fig_path, dpi=300)
        print(f"Successfully！Visual reesults saved to {output_fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"繪圖時發生錯誤：{e}")


if __name__ == '__main__':
    main()