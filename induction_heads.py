import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer

ROOT_PATH   = r"D:\new_representations\representations-in-tsfms-main\data"
DATA_FILE   = "ETTh1.csv"
SEQ_LEN     = 96
PRED_LEN    = 96
TOTAL_LEN   = SEQ_LEN + PRED_LEN
BATCH_SIZE  = 32
VOCAB_SIZE  = 50257
STRIDE      = 5

DATASET_NAME = "ETTh1"
FEATURES    = "S"
TARGET      = "OT"
FREQ        = "h"
LABEL_LEN   = 0
TRAIN_LIMIT = 5000
TEST_LIMIT  = 2000

torch.manual_seed(42)
np.random.seed(42)

from data_provider.data_factory import data_provider

DATASETS = {
    "ETTh1":      {"data": "ETTh1",      "data_path": "ETTh1.csv",      "features": "S", "target": "OT", "freq": "h", "use_all": False},
    "ETTh2":      {"data": "ETTh2",      "data_path": "ETTh2.csv",      "features": "S", "target": "OT", "freq": "h", "use_all": False},
    "ETTm1":      {"data": "ETTm1",      "data_path": "ETTm1.csv",      "features": "S", "target": "OT", "freq": "t", "use_all": True},
    "ETTm2":      {"data": "ETTm2",      "data_path": "ETTm2.csv",      "features": "S", "target": "OT", "freq": "t", "use_all": False},
    "traffic":    {"data": "custom",     "data_path": "traffic.csv",    "features": "M", "target": "OT", "freq": "h", "use_all": False},
    "weather":    {"data": "custom",     "data_path": "weather.csv",    "features": "M", "target": "OT", "freq": "h", "use_all": False},
    "electricity":{"data": "custom",     "data_path": "electricity.csv","features": "M", "target": "OT", "freq": "h", "use_all": False},
}

class DictObj:
    def __init__(self, in_dict: dict):
        self.__dict__.update(in_dict)
    def get(self, key, default=None): return self.__dict__.get(key, default)
    def __getitem__(self, key): return self.__dict__[key]

def load_dataset(dataset_name, ds_config, **kwargs):
    args_dict = {
        "root_path": ROOT_PATH,
        "data_path": ds_config["data_path"],
        "data": ds_config["data"],
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "features": ds_config["features"],
        "target": ds_config["target"],
        "freq": ds_config["freq"],
        "batch_size": BATCH_SIZE,
        "embed": "timeF",
        "num_workers": 0,
    }
    args = DictObj(args_dict)
    train_set, train_loader = data_provider(args, "train")
    test_set, test_loader   = data_provider(args, "test")

    def _build_tokens(loader, desc, use_all=False):
        all_data = []
        for batch in loader:
            seq_x = batch[0].numpy()
            for s in seq_x:
                if s.ndim > 1: s = s[:, 0]
                all_data.append(s)
        raw = np.concatenate(all_data, axis=0).astype(np.float32)
        d_min, d_max = raw.min(), raw.max()

        tokens = ((raw - d_min) / (d_max - d_min + 1e-8) * (VOCAB_SIZE - 1)).clip(0, VOCAB_SIZE - 1).astype(int)
        samples = [tokens[i:i + TOTAL_LEN] for i in range(0, len(tokens) - TOTAL_LEN + 1, STRIDE)]
        if not use_all:
            limit = TRAIN_LIMIT if "train" in desc else TEST_LIMIT
            samples = samples[:limit]
        return DataLoader(
            TensorDataset(torch.tensor(np.array(samples), dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=("train" in desc)
        ), d_min, d_max

    use_all = kwargs.get("use_all", False)
    train_loader, d_min, d_max = _build_tokens(train_loader, "train", use_all)
    test_loader, _, _          = _build_tokens(test_loader,  "test",  use_all)

    scaler    = test_set.scaler
    data_mean = float(scaler.mean_[0, 0] if scaler.mean_.ndim > 1 else scaler.mean_)
    data_std  = float(scaler.scale_[0, 0] if scaler.scale_.ndim > 1 else scaler.scale_)

    return train_loader, test_loader, d_min, d_max, data_mean, data_std


def to_continuous(tokens, d_min, d_max):
    norm = tokens.float() / (VOCAB_SIZE - 1)
    return norm * (d_max - d_min) + d_min

def build_cosine_bias(q_len, k_len, device, period=24.0, lam=1.0, phi=0.0):
    omega = 1.0 / period
    q_idx = torch.arange(q_len, device=device).float().unsqueeze(1)
    k_idx = torch.arange(k_len, device=device).float().unsqueeze(0)
    delta_t = torch.abs(q_idx - k_idx)
    return lam * torch.cos(2 * math.pi * omega * delta_t + phi)


def run_inference(model, data_loader, d_min, d_max, biased_heads=None, period=24.0, lam=1.0, phi=0.0):
    model.reset_hooks()
    hook_fns = []
    if biased_heads:
        def make_hook(layer, head):
            cache = {}
            def fn(attn_scores, hook):
                b, n_h, q_len, k_len = attn_scores.shape
                if q_len not in cache:
                    cache[q_len] = build_cosine_bias(q_len, k_len, model.cfg.device, period, lam, phi)
                attn_scores[:, head, :, :] += cache[q_len]
                return attn_scores
            return fn
        for (l, h) in biased_heads:
            model.add_hook(f"blocks.{l}.attn.hook_attn_scores", make_hook(l, h))

    total_mse, total_mae = 0.0, 0.0
    n_batches = 0
    sample_targets, sample_preds = None, None

    for batch in data_loader:
        batch = batch[0].to(model.cfg.device)   # [batch, 192]
        inputs  = batch[:, :-1]                  # [batch, 191] -> predict next token
        targets = batch[:, 1:]                   # [batch, 191]

        with torch.no_grad():
            logits = model(inputs)
            preds_tokens = torch.argmax(logits[:, -PRED_LEN:, :], dim=-1)  # [batch, 96]
            targets_slice = targets[:, -PRED_LEN:]  # [batch, 96]

            targets_cont = to_continuous(targets_slice, d_min, d_max)   # [batch, 96]
            preds_cont    = to_continuous(preds_tokens, d_min, d_max)    # [batch, 96]

            total_mse += F.mse_loss(preds_cont, targets_cont).item()
            total_mae += F.l1_loss(preds_cont, targets_cont).item()
            n_batches += 1

            if sample_targets is None:
                sample_targets = targets_cont[0].cpu().numpy()
                sample_preds   = preds_cont[0].cpu().numpy()

    model.reset_hooks()
    return total_mse / n_batches, total_mae / n_batches, sample_targets, sample_preds


def plot_heatmap_all(model, batch_tokens, n_layers, n_heads, title, highlighted=None,
                     biased_heads=None, period=24.0, lam=1.0, phi=0.0):
    model.reset_hooks()

    if biased_heads:
        def make_hook(layer, head):
            cache = {}
            def fn(attn_scores, hook):
                b, n_h, q_len, k_len = attn_scores.shape
                if q_len not in cache:
                    cache[q_len] = build_cosine_bias(q_len, k_len, model.cfg.device, period, lam, phi)
                attn_scores[:, head, :, :] += cache[q_len]
                return attn_scores
            return fn
        for (l, h) in biased_heads:
            model.add_hook(f"blocks.{l}.attn.hook_attn_scores", make_hook(l, h))

    _, cache = model.run_with_cache(batch_tokens, names_filter=lambda n: "pattern" in n)

    fig, axes = plt.subplots(n_layers, n_heads,
                              figsize=(n_heads * 1.8 + 1, n_layers * 1.8 + 1),
                              squeeze=False)
    fig.suptitle(title, fontsize=14)

    for l in range(n_layers):
        for h in range(n_heads):
            ax  = axes[l][h]
            attn = cache["pattern", l][0, h].cpu().numpy()
            ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=0.5)
            is_bias = highlighted and (l, h) in highlighted
            ax.set_title(f"L{l}H{h}" + (" ←" if is_bias else ""), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if is_bias:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()
    model.reset_hooks()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    n_layers = model.cfg.n_layers   # 12
    n_heads  = model.cfg.n_heads    # 12
    print(f"Model: GPT-2 Small | {n_layers} layers x {n_heads} heads")
    print("=" * 70)

    biased_heads = list(set([
        (0, 1), (0, 5), (0, 10), (1, 11), (3, 0),
        (5, 1), (5, 5), (6, 9), (7, 1), (7, 10),
        (8, 1), (8, 6), (9, 6), (9, 9), (10, 1),
        (10, 6), (11, 9), (11, 8), (10, 10), (10, 11),
    ]))
    lam = 1.0
    phi = 0.0

    header = f"{'Dataset':<14} {'Normal MSE':>12} {'Normal MAE':>12} {'Bias MSE':>12} {'Bias MAE':>12} {'MSE↓%':>8}"
    print(header)
    print("-" * 70)

    all_results = {}

    for ds_name, ds_config in DATASETS.items():
        print(f"\n[Loading {ds_name}...]")
        try:
            train_loader, test_loader, d_min, d_max, data_mean, data_std = load_dataset(ds_name, ds_config, use_all=ds_config.get("use_all", False))
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        print(f"  Test samples: {len(test_loader.dataset)}")

        mse_norm, mae_norm, _, _ = run_inference(model, test_loader, d_min, d_max)
        mse_bias, mae_bias, _, _ = run_inference(
            model, test_loader, d_min, d_max,
            biased_heads=biased_heads, period=24.0, lam=lam, phi=phi
        )

        mse_decrease = (mse_norm - mse_bias) / (mse_norm + 1e-8) * 100

        print(f"{ds_name:<14} {mse_norm:>12.4f} {mae_norm:>12.4f} {mse_bias:>12.4f} {mae_bias:>12.4f} {mse_decrease:>8.2f}%")

        all_results[ds_name] = {
            "mse_norm": mse_norm, "mae_norm": mae_norm,
            "mse_bias": mse_bias, "mae_bias": mae_bias,
            "mse_decrease": mse_decrease,
        }

    print("=" * 70)
    print("\nSummary (sorted by MSE decrease):")
    print("-" * 55)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mse_decrease"], reverse=True)
    for ds_name, r in sorted_results:
        marker = "✓" if r["mse_decrease"] > 0 else "✗"
        print(f"  {marker} {ds_name:<14}  MSE↓ {r['mse_decrease']:>6.2f}%  "
              f"({r['mse_norm']:.4f} → {r['mse_bias']:.4f})")


if __name__ == "__main__":
    main()
