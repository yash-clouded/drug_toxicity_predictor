"""
train_gnn.py
------------
Training script for the molecule Graph Neural Network.

Trains a separate GATConv model per Tox21 toxicity target.
Saves checkpoints to models/{target}_gnn.pt

Usage:
    python src/train_gnn.py
    python src/train_gnn.py --target SR-MMP --epochs 50
    python src/train_gnn.py --quick          # 1 target, smoke-test
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# pyg
try:
    from torch_geometric.loader import DataLoader as PyGLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[ERROR] PyTorch Geometric not found. Run: pip install torch-geometric")
    sys.exit(1)

from sklearn.metrics import roc_auc_score

# project imports (works when run as `python src/train_gnn.py` from project root)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gnn_model import ToxGNN, MoleculeDataset, train_epoch, eval_epoch, NODE_DIM
from data_processing import load_tox21, clean_dataset, get_binary_target, split_data, TOX21_TARGETS

BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "tox21.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Main training loop for a single target
# ---------------------------------------------------------------------------

def train_gnn_target(
    df,
    target: str,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.2,
    device: str = "cpu",
) -> dict:
    print(f"\n{'='*60}\n  GNN Target: {target}\n{'='*60}")

    smiles, labels = get_binary_target(df, target)
    splits = split_data(smiles, labels)

    smi_train, y_train = splits["train"]
    smi_val, y_val     = splits["val"]
    smi_test, y_test   = splits["test"]

    print(f"[gnn] Building datasets...")
    train_ds = MoleculeDataset(smi_train, y_train)
    val_ds   = MoleculeDataset(smi_val,   y_val)
    test_ds  = MoleculeDataset(smi_test,  y_test)
    print(f"[gnn] #train={len(train_ds)}  #val={len(val_ds)}  #test={len(test_ds)}")

    train_loader = PyGLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = PyGLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Handle class imbalance via pos_weight
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ToxGNN(
        in_channels=NODE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_auc = 0.0
    best_state   = None
    patience_cnt = 0
    early_stop_patience = 20

    for epoch in range(1, epochs + 1):
        # Use raw logits for BCEWithLogitsLoss by temporarily patching forward pass
        # We train with logits, eval with sigmoid
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Get raw logits (before final sigmoid) for BCEWithLogitsLoss
            # Temporarily access the pre-sigmoid output via the MLP last layer
            x, edge_index, b = batch.x, batch.edge_index, batch.batch
            from torch_geometric.nn import global_mean_pool, global_max_pool
            import torch.nn.functional as F

            x = F.relu(model.input_proj(x))
            x = F.dropout(x, p=model.dropout, training=True)
            for i, conv in enumerate(model.conv_layers):
                x_new = conv(x, edge_index)
                if i < len(model.conv_layers) - 1:
                    x_new = F.elu(x_new)
                    x_new = F.dropout(x_new, p=model.dropout, training=True)
                x = x_new
            x_mean = global_mean_pool(x, b)
            x_max  = global_max_pool(x, b)
            x      = torch.cat([x_mean, x_max], dim=1)
            # Get logits from all MLP layers except final sigmoid
            logits = model.mlp(x).squeeze(1)

            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)

        val_probs, val_labels = eval_epoch(model, val_loader, device)
        if len(np.unique(val_labels)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(val_labels, val_probs)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

        if patience_cnt >= early_stop_patience:
            print(f"[gnn] Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_probs, test_labels = eval_epoch(model, test_loader, device)
    test_auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5

    print(f"[gnn] Val ROC-AUC: {best_val_auc:.4f} | Test ROC-AUC: {test_auc:.4f}")

    # Save checkpoint and metadata
    safe_target = target.replace("-", "_")
    ckpt_path  = os.path.join(MODEL_DIR, f"{safe_target}_gnn.pt")
    meta_path  = os.path.join(MODEL_DIR, f"{safe_target}_gnn_meta.json")
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "target": target,
        "model_type": "GATConv",
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "heads": heads,
        "val_roc_auc": round(best_val_auc, 4),
        "test_roc_auc": round(test_auc, 4),
        "train_size": len(train_ds),
        "node_dim": NODE_DIM,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[gnn] Checkpoint saved: {ckpt_path}")
    return meta


# ---------------------------------------------------------------------------
# Multi-target training
# ---------------------------------------------------------------------------

def train_all_gnn(df, targets, **kwargs):
    all_metrics = []
    for target in targets:
        if target not in df.columns:
            print(f"[gnn] Skipping {target} — not in dataset.")
            continue
        try:
            m = train_gnn_target(df, target, **kwargs)
            all_metrics.append(m)
        except Exception as e:
            print(f"[gnn] ERROR on {target}: {e}")
            import traceback; traceback.print_exc()

    import pandas as pd
    if all_metrics:
        summary = pd.DataFrame(all_metrics)[["target", "val_roc_auc", "test_roc_auc", "train_size"]]
        path = os.path.join(RESULTS_DIR, "gnn_training_summary.csv")
        summary.to_csv(path, index=False)
        print(f"\n[gnn] GNN Summary saved to {path}")
        print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN toxicity models")
    parser.add_argument("--data",    default=DATA_PATH, help="Path to tox21.csv")
    parser.add_argument("--target",  default=None,      help="Single target (default: all)")
    parser.add_argument("--epochs",  type=int, default=60)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--hidden",  type=int, default=128)
    parser.add_argument("--layers",  type=int, default=3)
    parser.add_argument("--heads",   type=int, default=4)
    parser.add_argument("--quick",   action="store_true", help="Smoke test: 1 target, 5 epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[gnn] Using device: {device}")

    df = load_tox21(args.data)
    df = clean_dataset(df)

    if args.quick:
        targets = ["SR-MMP"]
        args.epochs = 5
    elif args.target:
        targets = [args.target]
    else:
        targets = TOX21_TARGETS

    train_all_gnn(
        df,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        heads=args.heads,
        device=device,
    )
