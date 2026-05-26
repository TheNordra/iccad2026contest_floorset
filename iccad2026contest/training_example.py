#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - SUPERVISED GNN Training (V2 architecture)
==========================================================================

Paradigm shift (2026-05-26): the contest is RECONSTRUCTION, not optimization.
See CLAUDE.md "範式轉移" section for the reasoning.

This script trains a GNN to PREDICT THE ORIGINAL FLOORPLAN (fp_sol) given the
inputs, instead of just minimising HPWL+overlap from scratch.

Key changes vs the v1 (unsupervised) script:
1. **Loss = MSE against fp_sol** (ground-truth (w, h, x, y) per block).
   The cost-function loss is still computed for monitoring but does not drive
   training.
2. **FloorplanNetV2 architecture**:
   - Wider features (14 dims, including boundary flags, preplaced, mib/cluster
     hints, pin count, log(area))
   - 4 residual GCN layers with LayerNorm + Dropout
   - No grid-position prior (interferes with supervised learning)
   - Single learnable scale: output xy ∈ [0, SCALE], SCALE=500
3. **Vectorised edge processing** using scatter_add — order-of-magnitude
   faster than the v1 Python edge loops.
4. **Saves to floorplan_gnn_v2.pth** — does NOT overwrite the v1 weights.
   optimizer_claude.py will be updated to prefer v2 over v1.

Run (sanity, ~5 min, no .pth side effects):
    python iccad2026contest/training_example.py --sanity

Run (full training, ~1.5h baseline @ 500 samples; 6-9h for 2000-3000):
    python iccad2026contest/training_example.py --num-samples 2000 --fresh
"""

import argparse
import math
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
# Architecture
# =========================================================================

class ResidualGCNLayer(nn.Module):
    """Pre-norm residual GCN block: x -> x + Dropout(ReLU(Linear(adj @ LN(x))))"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.matmul(adj, h)
        h = self.linear(h)
        h = torch.relu(h)
        h = self.dropout(h)
        return x + h


class FloorplanNetV2(nn.Module):
    """Supervised reconstruction GNN.

    Input features (per block, 14 dims):
        0  area
        1  sqrt(area)
        2  log(area + 1)
        3  avg pin x          (0 if no pins connected)
        4  avg pin y          (0 if no pins connected)
        5  pin_count
        6  log(pin_count + 1)
        7  is_boundary_left
        8  is_boundary_right
        9  is_boundary_top
        10 is_boundary_bottom
        11 is_preplaced
        12 is_fixed
        13 has_mib OR has_cluster (boolean: in any soft group)

    Output:
        For each block, (x, y, w, h):
            - (x, y) = sigmoid(pos_head(h)) * SCALE
            - (w, h) derived from area + tanh(ratio_head(h)) so that w*h == area
    """

    INPUT_DIM = 14
    DEFAULT_SCALE = 500.0
    LOG_RATIO_MAX = math.log(10.0)

    def __init__(self, hidden_dim: int = 256, n_gcn_layers: int = 4,
                 dropout: float = 0.1, scale: float = DEFAULT_SCALE):
        super().__init__()
        self.scale = scale
        self.input_proj = nn.Linear(self.INPUT_DIM, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [ResidualGCNLayer(hidden_dim, dropout=dropout)
             for _ in range(n_gcn_layers)]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.pos_head = nn.Linear(hidden_dim, 2)
        self.ratio_head = nn.Linear(hidden_dim, 1)

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _vec_pin_features(p2b_conn: torch.Tensor, pins_pos: torch.Tensor,
                          n: int, device) -> torch.Tensor:
        """Returns (n, 3) tensor: (avg_pin_x, avg_pin_y, pin_count) per block.
        Uses scatter_add — no Python loops over edges."""
        out = torch.zeros(n, 3, device=device)
        if p2b_conn is None or p2b_conn.dim() != 2 or p2b_conn.numel() == 0:
            return out
        valid = p2b_conn[:, 0] >= 0
        if not valid.any():
            return out
        pi = p2b_conn[valid, 0].long()
        bi = p2b_conn[valid, 1].long()
        in_range = (bi < n) & (pi < pins_pos.shape[0])
        pi = pi[in_range]
        bi = bi[in_range]
        if pi.numel() == 0:
            return out
        px = pins_pos[pi, 0]
        py = pins_pos[pi, 1]
        # scatter sums
        sum_x = torch.zeros(n, device=device)
        sum_y = torch.zeros(n, device=device)
        cnt   = torch.zeros(n, device=device)
        sum_x.scatter_add_(0, bi, px)
        sum_y.scatter_add_(0, bi, py)
        cnt.scatter_add_(0, bi, torch.ones_like(bi, dtype=torch.float, device=device))
        mask = cnt > 0
        out[mask, 0] = sum_x[mask] / cnt[mask]
        out[mask, 1] = sum_y[mask] / cnt[mask]
        out[:, 2] = cnt
        return out

    @staticmethod
    def _vec_adj(b2b_conn: torch.Tensor, n: int, device) -> torch.Tensor:
        """Returns (n, n) row-normalised adjacency. Vectorised over edges."""
        adj = torch.eye(n, device=device)
        if b2b_conn is None or b2b_conn.dim() != 2 or b2b_conn.numel() == 0:
            return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        valid = b2b_conn[:, 0] >= 0
        if not valid.any():
            return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        ei = b2b_conn[valid, 0].long()
        ej = b2b_conn[valid, 1].long()
        if b2b_conn.shape[1] > 2:
            ew = b2b_conn[valid, 2].float()
        else:
            ew = torch.ones(ei.shape[0], dtype=torch.float, device=device)
        in_range = (ei < n) & (ej < n) & (ei >= 0) & (ej >= 0)
        ei, ej, ew = ei[in_range], ej[in_range], ew[in_range]
        # Symmetric accumulation
        adj.index_put_((ei, ej), ew, accumulate=True)
        adj.index_put_((ej, ei), ew, accumulate=True)
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return adj

    # --- Forward ---------------------------------------------------------

    def forward(self, area_target: torch.Tensor, b2b_conn: torch.Tensor,
                p2b_conn: torch.Tensor, pins_pos: torch.Tensor,
                constraints: torch.Tensor, block_count: int) -> torch.Tensor:
        device = area_target.device
        n = block_count

        # ---- Build features --------------------------------------------
        feats = torch.zeros(n, self.INPUT_DIM, device=device)
        a = area_target[:n].clamp(min=1e-6)
        feats[:, 0] = a
        feats[:, 1] = torch.sqrt(a)
        feats[:, 2] = torch.log(a + 1.0)

        pin_feats = self._vec_pin_features(p2b_conn, pins_pos, n, device)
        feats[:, 3] = pin_feats[:, 0]
        feats[:, 4] = pin_feats[:, 1]
        feats[:, 5] = pin_feats[:, 2]
        feats[:, 6] = torch.log(pin_feats[:, 2] + 1.0)

        if constraints is not None and constraints.shape[0] >= n:
            cons = constraints[:n].long()
            bflag = cons[:, 4]
            feats[:, 7]  = ((bflag & 1) > 0).float()  # LEFT
            feats[:, 8]  = ((bflag & 2) > 0).float()  # RIGHT
            feats[:, 9]  = ((bflag & 4) > 0).float()  # TOP
            feats[:, 10] = ((bflag & 8) > 0).float()  # BOTTOM
            feats[:, 11] = (cons[:, 1] > 0).float()   # preplaced
            feats[:, 12] = (cons[:, 0] > 0).float()   # fixed
            feats[:, 13] = ((cons[:, 2] > 0) | (cons[:, 3] > 0)).float()

        # ---- Build adjacency -------------------------------------------
        adj = self._vec_adj(b2b_conn, n, device)

        # ---- Encoder ---------------------------------------------------
        x = self.input_proj(feats)
        x = self.input_norm(x)
        x = torch.relu(x)
        for layer in self.gcn_layers:
            x = layer(adj, x)
        x = self.head_norm(x)

        # ---- Heads -----------------------------------------------------
        xy = torch.sigmoid(self.pos_head(x)) * self.scale  # (n, 2)
        log_ratio = torch.tanh(self.ratio_head(x).squeeze(-1)) * self.LOG_RATIO_MAX
        ratio = torch.exp(log_ratio)
        w = torch.sqrt(a * ratio)
        h = torch.sqrt(a / ratio)
        wh = torch.stack([w, h], dim=1)
        return torch.cat([xy, wh], dim=1)


# =========================================================================
# Supervised loss
# =========================================================================

def supervised_loss(pred: torch.Tensor, fp_sol_b: torch.Tensor,
                    block_count: int):
    """MSE on (x, y, w, h) where fp_sol stores (w, h, x, y).

    Returns (total_loss, pos_mse, dim_mse) — all scalars.
    pred shape:    (n, 4)  = (x, y, w, h)
    fp_sol_b shape (max_n, 4) = (w, h, x, y); we only use first `block_count` rows.
    """
    n = block_count
    target = torch.empty_like(pred[:n])
    target[:, 0] = fp_sol_b[:n, 2]  # x
    target[:, 1] = fp_sol_b[:n, 3]  # y
    target[:, 2] = fp_sol_b[:n, 0]  # w
    target[:, 3] = fp_sol_b[:n, 1]  # h

    pos_mse = torch.mean((pred[:n, :2] - target[:, :2]) ** 2)
    dim_mse = torch.mean((pred[:n, 2:] - target[:, 2:]) ** 2)
    total = pos_mse + dim_mse
    return total, pos_mse.detach(), dim_mse.detach()


# =========================================================================
# Main
# =========================================================================

V2_FINAL_PATH      = "floorplan_gnn_v2.pth"
V2_CHECKPOINT_PATH = "floorplan_gnn_v2_checkpoint.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Supervised GNN training (V2) for ICCAD 2026 FloorSet.")
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity-check mode: 20 samples (= 5 batches), no checkpoint, "
             "no final .pth save. Validates the pipeline before a long run.")
    parser.add_argument(
        "--num-samples", type=int, default=None, metavar="N",
        help="Number of training samples to use. Default: 500 (~1.5h on "
             "RTX 3060 Ti). 2000 ~= 6h, 3000 ~= 9h. Overridden to 20 if "
             "--sanity is set unless explicitly provided.")
    parser.add_argument(
        "--fresh", action="store_true",
        help=f"Skip loading {V2_FINAL_PATH} - train from scratch. "
             "Recommended for long runs (>=2000 samples) so cosine LR "
             "starting at 0.001 does not destabilise loaded weights.")
    args = parser.parse_args()
    SANITY = args.sanity
    FRESH  = args.fresh

    if args.num_samples is not None:
        NUM_SAMPLES = args.num_samples
    elif SANITY:
        NUM_SAMPLES = 20
    else:
        NUM_SAMPLES = 500

    BATCH_SIZE = 4
    BASE_LR    = 0.001
    GRAD_CLIP  = 1.0

    print("="*70)
    title = "Supervised GNN Training (V2)"
    if SANITY:
        title += " [SANITY MODE]"
    print(f"ICCAD 2026 FloorSet - {title}")
    print("="*70)
    print(f"   num_samples = {NUM_SAMPLES}   fresh = {FRESH}   sanity = {SANITY}")
    print(f"   loss = MSE(prediction, fp_sol)  output = {V2_FINAL_PATH}")
    if SANITY:
        print(f"   SANITY: {NUM_SAMPLES} samples / no .pth save / no checkpoint")
    elif NUM_SAMPLES >= 1000:
        approx_hours = NUM_SAMPLES * 1.5 / 500
        print(f"   long training: estimated ~{approx_hours:.1f}h on RTX 3060 Ti")
        if not FRESH:
            print("   tip: add --fresh to avoid disturbing existing weights "
                  "with high starting LR")
    print("-"*70)

    print("\nLoading training data...")
    dataloader = get_training_dataloader(
        batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, shuffle=True)
    n_batches = len(dataloader)
    print(f"Loaded {n_batches} batches\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = FloorplanNetV2().to(device)

    if FRESH:
        print(f"[--fresh] Skipping load of {V2_FINAL_PATH}; training from scratch.")
    elif Path(V2_FINAL_PATH).exists():
        try:
            model.load_state_dict(torch.load(V2_FINAL_PATH, map_location=device))
            print(f"Loaded existing weights from {V2_FINAL_PATH}.")
        except Exception as e:
            print(f"WARNING: could not load {V2_FINAL_PATH} ({e}); "
                  f"training from scratch instead.")
    else:
        print(f"No existing {V2_FINAL_PATH}; training from random init.")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_batches), eta_min=BASE_LR * 0.01)

    model.train()

    for batch_idx, batch in enumerate(dataloader):
        (area_target, b2b_conn, p2b_conn, pins_pos, constraints,
         tree_sol, fp_sol, metrics) = batch

        current_batch_size = area_target.size(0)
        optimizer.zero_grad()

        total_loss = 0.0
        sum_pos_mse = 0.0
        sum_dim_mse = 0.0
        sum_unsup_cost = 0.0   # for monitoring only — not optimised

        for b in range(current_batch_size):
            b_area = area_target[b].to(device)
            b_b2b  = b2b_conn[b].to(device)
            b_p2b  = p2b_conn[b].to(device)
            b_pins = pins_pos[b].to(device)
            b_cons = constraints[b].to(device)
            b_metr = metrics[b].to(device)
            b_sol  = fp_sol[b].to(device)

            block_count = int((b_area != -1).sum().item())
            if block_count == 0:
                continue

            positions = model(b_area, b_b2b, b_p2b, b_pins, b_cons, block_count)

            loss, pos_mse, dim_mse = supervised_loss(positions, b_sol, block_count)
            total_loss = total_loss + loss
            sum_pos_mse += pos_mse.item()
            sum_dim_mse += dim_mse.item()

            # Diagnostic only: contest-formula cost (no grad path)
            with torch.no_grad():
                unsup = compute_training_loss_differentiable(
                    positions, b_b2b, b_p2b, b_pins,
                    b_area[:block_count], b_metr
                )
                sum_unsup_cost += unsup.item()

        total_loss = total_loss / max(1, current_batch_size)
        cur_lr = optimizer.param_groups[0]['lr']
        avg_pos = sum_pos_mse / max(1, current_batch_size)
        avg_dim = sum_dim_mse / max(1, current_batch_size)
        avg_cost = sum_unsup_cost / max(1, current_batch_size)
        print(f"Batch {batch_idx:>3d}  loss={total_loss.item():.4f}  "
              f"pos_mse={avg_pos:.2f}  dim_mse={avg_dim:.2f}  "
              f"unsup_cost={avg_cost:.3f}  lr={cur_lr:.5f}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if not SANITY and batch_idx > 0 and batch_idx % 20 == 0:
            torch.save(model.state_dict(), V2_CHECKPOINT_PATH)
            print(f"[Checkpoint] weights -> {V2_CHECKPOINT_PATH} "
                  f"(batch {batch_idx})")

    print("\n" + "="*70)
    if SANITY:
        print("[SANITY] Training loop finished. SKIPPING .pth save.")
        print("[SANITY] Check above: loss / pos_mse / dim_mse should DECREASE")
        print("[SANITY] and unsup_cost should NOT spike (proxy for inference quality).")
        print("[SANITY] If healthy, drop --sanity and rerun for real training.")
    else:
        print("Training loop finished successfully!")
        torch.save(model.state_dict(), V2_FINAL_PATH)
        print(f"[Final] weights saved to {V2_FINAL_PATH}")
    print("="*70)

    # ---- Visualisation (also a sanity check that inference shape works) ----
    print("\nGenerating visualisation...")
    model.eval()
    with torch.no_grad():
        s_area = area_target[0].to(device)
        s_b2b  = b2b_conn[0].to(device)
        s_p2b  = p2b_conn[0].to(device)
        s_pins = pins_pos[0].to(device)
        s_cons = constraints[0].to(device)
        s_sol  = fp_sol[0].to(device)
        s_metr = metrics[0].to(device)
        bc = int((s_area != -1).sum().item())
        pred = model(s_area, s_b2b, s_p2b, s_pins, s_cons, bc)
        sl, p_mse, d_mse = supervised_loss(pred, s_sol, bc)

        pred_np = pred.cpu().numpy()
        sol_np = s_sol[:bc].cpu().numpy()
        pins_np = s_pins.cpu().numpy()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        # Predicted
        for i in range(bc):
            x, y, w, h = pred_np[i]
            axes[0].add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='blue',
                facecolor='skyblue', alpha=0.4))
            axes[0].text(x + w/2, y + h/2, str(i), fontsize=6,
                         ha='center', va='center')
        # Ground truth
        for i in range(bc):
            w, h, x, y = sol_np[i]
            axes[1].add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='green',
                facecolor='lightgreen', alpha=0.4))
            axes[1].text(x + w/2, y + h/2, str(i), fontsize=6,
                         ha='center', va='center')
        # Pins on both
        if len(pins_np) > 0:
            valid = pins_np[pins_np[:, 0] != -1]
            if len(valid) > 0:
                for ax in axes:
                    ax.scatter(valid[:, 0], valid[:, 1], c='red', s=15,
                               marker='x', label='Pins')
        for ax, title in zip(axes,
                             [f"Predicted (pos_mse={p_mse.item():.2f} "
                              f"dim_mse={d_mse.item():.2f})",
                              "Ground truth (fp_sol)"]):
            ax.set_xlim(-10, 510)
            ax.set_ylim(-10, 510)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.legend(loc='upper right')
        plt.suptitle(f"FloorplanNetV2 — sample 0, n={bc} blocks")
        out_path = "predicted_floorplan_v2.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()
