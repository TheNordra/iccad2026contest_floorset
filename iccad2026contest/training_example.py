#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - STRUCTURAL GNN Training (V3 architecture)
==========================================================================

Path C pivot (2026-05-27): predict STRUCTURE, not absolute positions.

V2 (supervised MSE on fp_sol positions) failed: the task is ill-posed
one-to-many (many valid layouts satisfy the same inputs), so MSE drove
predictions toward an unphysical "mean layout" — pos_mse oscillated and
unsup_cost exploded into the millions. See CLAUDE.md for the full diagnosis.

V3 idea: the skyline BL packer only needs a PERMUTATION (which block comes
first), not absolute positions. So we train the GNN to output, per block, a
single "BL ordering score". The training signal is fp_sol's `(x + y)` —
blocks nearer the lower-left should rank lower. The pairwise ranking loss
ignores absolute scale and only enforces ordering, which is well-posed even
under one-to-many.

Architecture is identical to v2 (residual GCN + LayerNorm + 14-dim features +
vectorised scatter_add). Heads change:
  v2: pos_head(2) + ratio_head(1)
  v3: bl_head(1)  + ratio_head(1)   ← bl_head replaces pos_head

Output of v3 forward(): (block_count, 4) for backward compat with v1/v2
inference code paths, packed as:
    [bl_score, 0, w, h]
       ^^^^^^^^         ← BL score in column 0; column 1 ignored
                  ^^^^^ ← aspect ratio derived from area + tanh ratio_head

This way `optimizer_claude.py` can sort blocks by `pred[:, 0]` (or `cx+cy`,
since the second column is 0 — same ordering). Once optimizer_claude.py is
updated to be v3-aware, it'll use the score directly.

Saves to floorplan_gnn_v3.pth — does NOT overwrite v1 or v2.

Run (sanity, ~5 min, no .pth side effects):
    python iccad2026contest/training_example.py --sanity

Run (full training, ~3h @ 2000 samples with scatter_add vectorisation):
    python iccad2026contest/training_example.py --num-samples 2000 --fresh
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)


# =========================================================================
# Architecture
# =========================================================================

class ResidualGCNLayer(nn.Module):
    """Pre-norm residual GCN: x -> x + Dropout(ReLU(Linear(adj @ LN(x))))"""

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


class FloorplanNetV3(nn.Module):
    """Structural GNN: per-block BL-ordering score + aux aspect ratio.

    Input features (14 dims, same as v2):
        0  area
        1  sqrt(area)
        2  log(area + 1)
        3  avg pin x
        4  avg pin y
        5  pin count
        6  log(pin count + 1)
        7  is_boundary_left
        8  is_boundary_right
        9  is_boundary_top
        10 is_boundary_bottom
        11 is_preplaced
        12 is_fixed
        13 has_mib OR has_cluster

    Output: (block_count, 4)
        column 0: BL score (sort key for permutation)
        column 1: 0.0 (placeholder, kept for shape compat with v1/v2)
        column 2: w   (derived from area * exp(tanh(ratio)*log(10)))
        column 3: h   (= area / w)

    The shape-compat output lets optimizer_claude.py continue using
    `pred[:, 0] + pred[:, 1]` as a sort key (= bl_score + 0 = bl_score) until
    it is updated for v3 awareness.
    """

    INPUT_DIM = 14
    LOG_RATIO_MAX = math.log(10.0)

    def __init__(self, hidden_dim: int = 256, n_gcn_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(self.INPUT_DIM, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [ResidualGCNLayer(hidden_dim, dropout=dropout)
             for _ in range(n_gcn_layers)]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.bl_head = nn.Linear(hidden_dim, 1)
        self.ratio_head = nn.Linear(hidden_dim, 1)

    # --- Vectorised helpers (same as v2) -------------------------------

    @staticmethod
    def _vec_pin_features(p2b_conn, pins_pos, n, device):
        out = torch.zeros(n, 3, device=device)
        if p2b_conn is None or p2b_conn.dim() != 2 or p2b_conn.numel() == 0:
            return out
        valid = p2b_conn[:, 0] >= 0
        if not valid.any():
            return out
        pi = p2b_conn[valid, 0].long()
        bi = p2b_conn[valid, 1].long()
        in_range = (bi < n) & (pi < pins_pos.shape[0])
        pi, bi = pi[in_range], bi[in_range]
        if pi.numel() == 0:
            return out
        sum_x = torch.zeros(n, device=device)
        sum_y = torch.zeros(n, device=device)
        cnt   = torch.zeros(n, device=device)
        sum_x.scatter_add_(0, bi, pins_pos[pi, 0])
        sum_y.scatter_add_(0, bi, pins_pos[pi, 1])
        cnt.scatter_add_(0, bi, torch.ones_like(bi, dtype=torch.float, device=device))
        mask = cnt > 0
        out[mask, 0] = sum_x[mask] / cnt[mask]
        out[mask, 1] = sum_y[mask] / cnt[mask]
        out[:, 2] = cnt
        return out

    @staticmethod
    def _vec_adj(b2b_conn, n, device):
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
        adj.index_put_((ei, ej), ew, accumulate=True)
        adj.index_put_((ej, ei), ew, accumulate=True)
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return adj

    def forward(self, area_target, b2b_conn, p2b_conn, pins_pos,
                constraints, block_count):
        device = area_target.device
        n = block_count

        feats = torch.zeros(n, self.INPUT_DIM, device=device)
        a = area_target[:n].clamp(min=1e-6)
        feats[:, 0] = a
        feats[:, 1] = torch.sqrt(a)
        feats[:, 2] = torch.log(a + 1.0)

        pin_feats = self._vec_pin_features(p2b_conn, pins_pos, n, device)
        feats[:, 3:6] = pin_feats
        feats[:, 6] = torch.log(pin_feats[:, 2] + 1.0)

        if constraints is not None and constraints.shape[0] >= n:
            cons = constraints[:n].long()
            bflag = cons[:, 4]
            feats[:, 7]  = ((bflag & 1) > 0).float()
            feats[:, 8]  = ((bflag & 2) > 0).float()
            feats[:, 9]  = ((bflag & 4) > 0).float()
            feats[:, 10] = ((bflag & 8) > 0).float()
            feats[:, 11] = (cons[:, 1] > 0).float()
            feats[:, 12] = (cons[:, 0] > 0).float()
            feats[:, 13] = ((cons[:, 2] > 0) | (cons[:, 3] > 0)).float()

        adj = self._vec_adj(b2b_conn, n, device)

        x = self.input_proj(feats)
        x = self.input_norm(x)
        x = torch.relu(x)
        for layer in self.gcn_layers:
            x = layer(adj, x)
        x = self.head_norm(x)

        # Heads
        bl = self.bl_head(x).squeeze(-1)   # (n,) raw scores (any real value)
        log_ratio = torch.tanh(self.ratio_head(x).squeeze(-1)) * self.LOG_RATIO_MAX
        ratio = torch.exp(log_ratio)
        w = torch.sqrt(a * ratio)
        h = torch.sqrt(a / ratio)

        # Pack into (n, 4) for back-compat shape: [bl, 0, w, h]
        zero_pad = torch.zeros_like(bl)
        return torch.stack([bl, zero_pad, w, h], dim=1)


# =========================================================================
# Loss
# =========================================================================

def pairwise_ranking_loss(bl_score: torch.Tensor, gt_score: torch.Tensor,
                          block_count: int) -> torch.Tensor:
    """
    Pairwise BCE: for all (i, j) with i != j, want sign(bl[i] - bl[j]) ==
    sign(gt[i] - gt[j]). Implemented as BCE on logits = bl[i] - bl[j],
    target = 1 if gt[i] > gt[j].

    With block_count = n, this is O(n^2) pairs (~7k for n=120). For batch
    of 4, ~28k pairs per backward — fast on GPU.

    Returns scalar loss.
    """
    n = block_count
    bl = bl_score[:n]
    gt = gt_score[:n]
    # All-pairs differences
    diff_pred = bl.unsqueeze(1) - bl.unsqueeze(0)  # (n, n): bl[i] - bl[j]
    diff_gt   = gt.unsqueeze(1) - gt.unsqueeze(0)  # (n, n): gt[i] - gt[j]
    target = (diff_gt > 0).float()                  # 1 if i should rank after j
    # Mask out diagonal (i == j has no signal)
    mask = ~torch.eye(n, dtype=torch.bool, device=bl.device)
    logits = diff_pred[mask]
    labels = target[mask]
    return F.binary_cross_entropy_with_logits(logits, labels)


def aspect_loss(pred: torch.Tensor, fp_sol_b: torch.Tensor,
                block_count: int) -> torch.Tensor:
    """MSE on (w, h) — kept from v2 since aspect ratio prediction worked there."""
    n = block_count
    w_pred = pred[:n, 2]
    h_pred = pred[:n, 3]
    w_true = fp_sol_b[:n, 0]
    h_true = fp_sol_b[:n, 1]
    return torch.mean((w_pred - w_true) ** 2) + torch.mean((h_pred - h_true) ** 2)


def ranking_accuracy(bl_score: torch.Tensor, gt_score: torch.Tensor,
                     block_count: int) -> float:
    """Fraction of pairs (i, j) where predicted and true orderings agree.
    Diagnostic only — no gradient. Range [0.5, 1.0]; 1.0 = perfect ranking."""
    n = block_count
    with torch.no_grad():
        bl = bl_score[:n]
        gt = gt_score[:n]
        diff_pred = bl.unsqueeze(1) - bl.unsqueeze(0)
        diff_gt   = gt.unsqueeze(1) - gt.unsqueeze(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=bl.device)
        agree = (diff_pred[mask].sign() == diff_gt[mask].sign())
        # Exclude exact ties (where diff_gt == 0) from denominator
        non_tie = diff_gt[mask] != 0
        if not non_tie.any():
            return 1.0
        return float(agree[non_tie].float().mean().item())


# =========================================================================
# Main
# =========================================================================

V3_FINAL_PATH      = "floorplan_gnn_v3.pth"
V3_CHECKPOINT_PATH = "floorplan_gnn_v3_checkpoint.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Structural GNN training (V3) for ICCAD 2026 FloorSet. "
                    "Predicts per-block BL-ordering scores via pairwise "
                    "ranking on fp_sol's (x+y).")
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity-check mode: 20 samples (= 5 batches), no checkpoint, "
             "no final .pth save. Validates the pipeline before long runs.")
    parser.add_argument(
        "--num-samples", type=int, default=None, metavar="N",
        help="Number of training samples to use. Default: 500 (~30 min @ "
             "RTX 3060 Ti with scatter_add). 2000 ~= 2-3h, 3000 ~= 4-5h. "
             "Overridden to 20 if --sanity is set unless explicitly provided.")
    parser.add_argument(
        "--fresh", action="store_true",
        help=f"Skip loading {V3_FINAL_PATH} - train from scratch. "
             "Recommended for long runs (>=2000 samples).")
    parser.add_argument(
        "--aspect-weight", type=float, default=0.01, metavar="LAMBDA",
        help="Weight for auxiliary aspect-ratio MSE loss. Default: 0.01. "
             "Set 0 to disable aspect supervision (BL ranking only).")
    args = parser.parse_args()
    SANITY = args.sanity
    FRESH  = args.fresh
    LAMBDA_ASPECT = args.aspect_weight

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
    title = "Structural GNN Training (V3 - BL ranking)"
    if SANITY:
        title += " [SANITY MODE]"
    print(f"ICCAD 2026 FloorSet - {title}")
    print("="*70)
    print(f"   num_samples = {NUM_SAMPLES}   fresh = {FRESH}   sanity = {SANITY}")
    print(f"   loss = pairwise_ranking(BL, fp_sol.x+y) "
          f"+ {LAMBDA_ASPECT} * MSE(w,h)")
    print(f"   output: {V3_FINAL_PATH}")
    if SANITY:
        print(f"   SANITY: {NUM_SAMPLES} samples / no .pth save / no checkpoint")
    elif NUM_SAMPLES >= 1000:
        approx_hours = NUM_SAMPLES * 2.5 / 2000  # scatter_add speed assumption
        print(f"   long training: estimated ~{approx_hours:.1f}h on RTX 3060 Ti")
    print("-"*70)

    print("\nLoading training data...")
    dataloader = get_training_dataloader(
        batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, shuffle=True)
    n_batches = len(dataloader)
    print(f"Loaded {n_batches} batches\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = FloorplanNetV3().to(device)

    if FRESH:
        print(f"[--fresh] Skipping load of {V3_FINAL_PATH}; training from scratch.")
    elif Path(V3_FINAL_PATH).exists():
        try:
            model.load_state_dict(torch.load(V3_FINAL_PATH, map_location=device))
            print(f"Loaded existing weights from {V3_FINAL_PATH}.")
        except Exception as e:
            print(f"WARNING: could not load {V3_FINAL_PATH} ({e}); "
                  f"training from scratch.")
    else:
        print(f"No existing {V3_FINAL_PATH}; training from random init.")

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
        sum_rank_loss = 0.0
        sum_aspect_loss = 0.0
        sum_rank_acc = 0.0
        sum_unsup = 0.0
        n_processed = 0

        for b in range(current_batch_size):
            b_area = area_target[b].to(device)
            b_b2b  = b2b_conn[b].to(device)
            b_p2b  = p2b_conn[b].to(device)
            b_pins = pins_pos[b].to(device)
            b_cons = constraints[b].to(device)
            b_sol  = fp_sol[b].to(device)
            b_metr = metrics[b].to(device)

            block_count = int((b_area != -1).sum().item())
            if block_count < 2:
                continue  # ranking needs >= 2 blocks

            pred = model(b_area, b_b2b, b_p2b, b_pins, b_cons, block_count)
            # pred = [bl, 0, w, h]
            bl_score = pred[:block_count, 0]

            # Ground-truth BL ordering signal: fp_sol's (x + y) per block
            gt_bl = b_sol[:block_count, 2] + b_sol[:block_count, 3]

            r_loss = pairwise_ranking_loss(bl_score, gt_bl, block_count)
            a_loss = aspect_loss(pred, b_sol, block_count)
            loss = r_loss + LAMBDA_ASPECT * a_loss

            total_loss = total_loss + loss
            sum_rank_loss += r_loss.item()
            sum_aspect_loss += a_loss.item()
            sum_rank_acc += ranking_accuracy(bl_score, gt_bl, block_count)
            n_processed += 1

            # Diagnostic: build a position tensor from BL score + (w, h) and
            # run through contest cost function. The "positions" we hand it
            # are NOT meant to be a real layout — sorting blocks by bl_score
            # and applying the BL packer is what optimizer_claude does at
            # inference. Here we just sanity-check that the contest cost
            # doesn't explode on the raw bl_score (no NaN, no inf).
            with torch.no_grad():
                # Build a pseudo layout: x = bl_score (rescaled), y = 0
                # — purely a sanity probe; do not interpret as a quality signal
                rescale = (bl_score - bl_score.min()).clamp(min=1e-6)
                rescale = rescale / rescale.max() * 100.0
                pseudo_pos = torch.zeros(block_count, 4, device=device)
                pseudo_pos[:, 0] = rescale
                pseudo_pos[:, 2] = pred[:block_count, 2]
                pseudo_pos[:, 3] = pred[:block_count, 3]
                try:
                    u = compute_training_loss_differentiable(
                        pseudo_pos, b_b2b, b_p2b, b_pins,
                        b_area[:block_count], b_metr)
                    sum_unsup += u.item()
                except Exception:
                    pass  # ignore probe failures

        if n_processed == 0:
            continue

        total_loss = total_loss / n_processed
        cur_lr = optimizer.param_groups[0]['lr']
        avg_rank = sum_rank_loss / n_processed
        avg_asp  = sum_aspect_loss / n_processed
        avg_acc  = sum_rank_acc / n_processed
        avg_uns  = sum_unsup / n_processed
        print(f"Batch {batch_idx:>3d}  loss={total_loss.item():.4f}  "
              f"rank={avg_rank:.4f}  aspect={avg_asp:.2f}  "
              f"rank_acc={avg_acc:.3f}  "
              f"probe_unsup={avg_uns:.2f}  lr={cur_lr:.5f}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if not SANITY and batch_idx > 0 and batch_idx % 20 == 0:
            torch.save(model.state_dict(), V3_CHECKPOINT_PATH)
            print(f"[Checkpoint] weights -> {V3_CHECKPOINT_PATH} "
                  f"(batch {batch_idx})")

    print("\n" + "="*70)
    if SANITY:
        print("[SANITY] Training loop finished. SKIPPING .pth save.")
        print("[SANITY] Key signal to check: rank_acc should INCREASE from")
        print("[SANITY] ~0.5 (random) toward 1.0 (perfect ranking).")
        print("[SANITY] rank loss should DECREASE; aspect loss decreasing is a bonus.")
        print("[SANITY] If healthy, drop --sanity and rerun for real training.")
    else:
        print("Training loop finished successfully!")
        torch.save(model.state_dict(), V3_FINAL_PATH)
        print(f"[Final] weights saved to {V3_FINAL_PATH}")
    print("="*70)

    # ---- Visualisation: predicted BL ordering vs ground truth ----
    print("\nGenerating visualisation...")
    model.eval()
    with torch.no_grad():
        s_area = area_target[0].to(device)
        s_b2b  = b2b_conn[0].to(device)
        s_p2b  = p2b_conn[0].to(device)
        s_pins = pins_pos[0].to(device)
        s_cons = constraints[0].to(device)
        s_sol  = fp_sol[0].to(device)
        bc = int((s_area != -1).sum().item())
        pred = model(s_area, s_b2b, s_p2b, s_pins, s_cons, bc)
        bl_pred = pred[:bc, 0]
        gt_bl = s_sol[:bc, 2] + s_sol[:bc, 3]
        acc = ranking_accuracy(bl_pred, gt_bl, bc)

        # Sort indices by predicted BL score (this is the perm v3 will output)
        pred_order = torch.argsort(bl_pred).cpu().numpy()
        gt_order = torch.argsort(gt_bl).cpu().numpy()

        bl_pred_np = bl_pred.cpu().numpy()
        gt_bl_np = gt_bl.cpu().numpy()
        sol_np = s_sol[:bc].cpu().numpy()

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # (1) Scatter: predicted bl vs ground-truth (x+y)
        axes[0].scatter(gt_bl_np, bl_pred_np, alpha=0.5)
        axes[0].set_xlabel("Ground-truth (x + y)")
        axes[0].set_ylabel("Predicted BL score")
        axes[0].set_title(f"Score correlation (rank_acc={acc:.3f})")
        axes[0].grid(True, alpha=0.3)

        # (2) Rank vs rank (perfect line = perfect ranking)
        pred_ranks = bl_pred.argsort().argsort().cpu().numpy()
        gt_ranks   = gt_bl.argsort().argsort().cpu().numpy()
        axes[1].scatter(gt_ranks, pred_ranks, alpha=0.5)
        axes[1].plot([0, bc], [0, bc], 'r--', alpha=0.5)
        axes[1].set_xlabel("Ground-truth rank")
        axes[1].set_ylabel("Predicted rank")
        axes[1].set_title("Rank-vs-rank")
        axes[1].grid(True, alpha=0.3)

        # (3) Ground truth layout coloured by predicted rank
        import matplotlib.patches as patches
        for i in range(bc):
            w, h, x, y = sol_np[i]
            color = plt.cm.viridis(pred_ranks[i] / max(1, bc - 1))
            axes[2].add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=0.5, edgecolor='black',
                facecolor=color, alpha=0.7))
        axes[2].set_xlim(-10, sol_np[:, 2].max() + sol_np[:, 0].max() + 10)
        axes[2].set_ylim(-10, sol_np[:, 3].max() + sol_np[:, 1].max() + 10)
        axes[2].set_aspect('equal')
        axes[2].set_title("fp_sol coloured by predicted BL rank")

        plt.suptitle(f"FloorplanNetV3 - sample 0, n={bc} blocks")
        out_path = "predicted_floorplan_v3.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()
