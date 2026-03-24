import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.loading_from_pretrained as loading
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data_provider.data_factory import data_provider
from data_provider.data_loader import tokenize_data


class DictObj:
    def __init__(self, in_dict: dict):
        self.__dict__.update(in_dict)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__


ROOT_PATH = r"D:\new_representations\representations-in-tsfms-main\data"
MODEL_PATH = r"D:\new_representations\representations-in-tsfms-main\output\custom-gpt2-4l4h\run-3\checkpoint-final"

SEQ_LEN = 96
PRED_LEN = 96
TOTAL_LEN = SEQ_LEN + PRED_LEN
BATCH_SIZE = 64
VOCAB_SIZE = 4096

EXPERIENCE_PERIOD = 24.0
LAMBDA = 0.3
SIGMA = 5.0

STRIDE = 5

torch.manual_seed(42)
np.random.seed(42)

DATASET_NAME = "ETTh1"
DATA_FILE = "ETTh1.csv"
FEATURES = "S"
TARGET = "OT"
FREQ = "h"
LABEL_LEN = 0


def load_and_prepare_data():
    args_dict = {
        "root_path": ROOT_PATH,
        "data_path": DATA_FILE,
        "data": DATASET_NAME,
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "features": FEATURES,
        "target": TARGET,
        "freq": FREQ,
        "batch_size": BATCH_SIZE,
        "embed": "timeF",
        "num_workers": 0,
    }
    args = DictObj(args_dict)

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")

    def _tokens_from_loader(data_set, loader, desc=""):
        all_data = []
        for batch in loader:
            seq_x = batch[0].numpy()
            for sample in seq_x:
                if sample.ndim > 1:
                    sample = sample[:, 0]
                all_data.append(sample)
        raw_values = np.concatenate(all_data, axis=0).astype(np.float32)

        tokens, data_min, data_max = tokenize_data(raw_values, VOCAB_SIZE)

        samples = []
        for i in range(0, len(tokens) - TOTAL_LEN + 1, STRIDE):
            samples.append(tokens[i: i + TOTAL_LEN])

        max_limit = {"train": 2000, "val": 400, "Test": 800}
        key = desc.split("_")[0].capitalize()
        samples = samples[:max_limit.get(key, len(samples))]

        dataset = TensorDataset(torch.tensor(np.array(samples), dtype=torch.long))
        shuffle = True if desc.startswith("train") else False
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle), data_min, data_max

    train_loader, d_min, d_max = _tokens_from_loader(train_set, train_loader, "train")
    val_loader, _, _ = _tokens_from_loader(val_set, val_loader, "val")
    test_loader, _, _ = _tokens_from_loader(test_set, test_loader, "test")

    scaler = test_set.scaler
    data_mean = scaler.mean_[0] if scaler.mean_.ndim > 0 else scaler.mean_
    data_std = scaler.scale_[0] if scaler.scale_.ndim > 0 else scaler.scale_

    return train_loader, val_loader, test_loader, float(d_min), float(d_max), float(data_mean), float(data_std)


def tokens_to_continuous(tokens, d_min, d_max):
    norm_val = tokens.float() / (VOCAB_SIZE - 1)
    normalized_data = (norm_val * 2.0) - 1.0
    return (normalized_data + 1.0) / 2.0 * (d_max - d_min) + d_min


def find_induction_heads(model, tokens_orig, tokens_shuf, n_layers, n_heads, target_idx, device):
    labels = np.array([1] * tokens_orig.size(0) + [0] * tokens_shuf.size(0))
    all_tokens = torch.cat([tokens_orig, tokens_shuf], dim=0)

    batch_size = 32
    accum_results = None
    for i in range(0, all_tokens.size(0), batch_size):
        batch = all_tokens[i:i + batch_size].to(device)
        _, cache = model.run_with_cache(batch)
        for l in range(1, n_layers):
            layer_results = cache["result", l][:, target_idx, :, :].cpu()
            if accum_results is None:
                accum_results = {}
                for ll in range(1, n_layers):
                    accum_results[ll] = []
            accum_results[l].append(layer_results)
        model.reset_hooks()

    all_heads = {}
    for l in range(1, n_layers):
        layer_tensor = torch.cat(accum_results[l], dim=0)
        best_ldr, best_h = -1.0, -1
        for h in range(n_heads):
            X = layer_tensor[:, h, :].numpy()
            y = labels
            lda = LinearDiscriminantAnalysis(n_components=1)
            try:
                X_proj = lda.fit_transform(X, y).flatten()
                mu_s, mu_c = X_proj[y == 1].mean(), X_proj[y == 0].mean()
                var_s, var_c = X_proj[y == 1].var(), X_proj[y == 0].var()
                ldr_score = ((mu_s - mu_c) ** 2) / (var_s + var_c + 1e-8)
            except:
                ldr_score = 0.0
            all_heads[(l, h)] = ldr_score
            if ldr_score > best_ldr:
                best_ldr, best_h = ldr_score, h
        print(f"   Layer {l} Head {best_h} LDR={best_ldr:.4f}")

    top_head = max(all_heads, key=all_heads.get)
    print(f"   Top Induction Head: Layer{top_head[0]} Head{top_head[1]} LDR={all_heads[top_head]:.4f}")

    _, ref_cache = model.run_with_cache(tokens_orig[:1].to(device))
    return top_head, all_heads, ref_cache



def make_cosine_bias_hook(omega, lam, device, target_head=None):
    def hook_fn(attn_scores, hook):
        b, n_h, sq, sk = attn_scores.shape
        q_idx = torch.arange(sq, device=device).unsqueeze(1)
        k_idx = torch.arange(sk, device=device).unsqueeze(0)
        delta_t = torch.abs(q_idx - k_idx).float()
        bias = lam * torch.cos(2 * math.pi * omega * delta_t)

        if target_head is not None and target_head != "all":
            attn_scores[:, target_head, :, :] = attn_scores[:, target_head, :, :] + bias
        else:
            attn_scores = attn_scores + bias.unsqueeze(0).unsqueeze(0)
        return attn_scores

    return hook_fn


def make_exp_decay_bias_hook(sigma, lam, device, target_head=None):
    def hook_fn(attn_scores, hook):
        b, n_h, sq, sk = attn_scores.shape
        q_idx = torch.arange(sq, device=device).unsqueeze(1)
        k_idx = torch.arange(sk, device=device).unsqueeze(0)
        delta_t = torch.abs(q_idx - k_idx).float()
        bias = lam * torch.exp(-delta_t / sigma)

        if target_head is not None and target_head != "all":
            attn_scores[:, target_head, :, :] = attn_scores[:, target_head, :, :] + bias
        else:
            attn_scores = attn_scores + bias.unsqueeze(0).unsqueeze(0)
        return attn_scores

    return hook_fn


def make_patch_hook(layer, target_head, ref_cache):
    def hook_fn(value, hook):
        b, seq_len, n_h, d = value.shape
        if target_head is not None and target_head != "all":
            ref_val = ref_cache["result", layer][0, :, target_head, :]  # [ref_seq_len, d_model]
            if ref_val.shape[0] >= seq_len:
                ref_val = ref_val[:seq_len, :]
            else:
                pad = torch.zeros(seq_len - ref_val.shape[0], d, device=value.device, dtype=value.dtype)
                ref_val = torch.cat([ref_val, pad], dim=0)
            value[:, :, target_head, :] = ref_val.unsqueeze(0).expand(b, -1, -1)
        else:
            ref_val = ref_cache["result", layer][0, :, :, :]  # [ref_seq_len, n_heads, d_model]
            if ref_val.shape[0] >= seq_len:
                ref_val = ref_val[:seq_len, :, :]
            else:
                pad = torch.zeros(seq_len - ref_val.shape[0], n_h, d, device=value.device, dtype=value.dtype)
                ref_val = torch.cat([ref_val, pad], dim=0)
            value = ref_val.unsqueeze(0).expand(b, -1, -1, -1)
        return value

    return hook_fn


def run_eval(model, test_loader, data_min, data_max, data_mean, data_std,
             method, param, device, omega, ref_cache):
    total_mse, total_mae = 0.0, 0.0
    count = 0


    if param == "all":
        target_layers = range(model.cfg.n_layers)
        target_head = "all"
    elif param is not None:
        target_layers = [param[0]]
        target_head = param[1]
    else:
        target_layers = []
        target_head = None

    for batch in tqdm(test_loader, desc="Test", leave=False):
        batch_data = batch[0].to(device)
        inputs = batch_data[:, :-1]
        targets = batch_data[:, 1:]

        # 挂载钩子
        for l in target_layers:
            if method == "cosine":
                model.add_hook(
                    f"blocks.{l}.attn.hook_attn_scores",
                    make_cosine_bias_hook(omega, LAMBDA, device, target_head)
                )
            elif method == "exp_decay":
                model.add_hook(
                    f"blocks.{l}.attn.hook_attn_scores",
                    make_exp_decay_bias_hook(SIGMA, LAMBDA, device, target_head)
                )
            elif method == "patch":
                model.add_hook(
                    f"blocks.{l}.attn.hook_result",
                    make_patch_hook(l, target_head, ref_cache)
                )

        with torch.no_grad():
            logits = model(inputs)
            preds_tokens = torch.argmax(logits[:, -1:, :], dim=-1)

            targets_cont = tokens_to_continuous(targets[:, -1:], data_min, data_max)
            preds_cont = tokens_to_continuous(preds_tokens, data_min, data_max)

            total_mse += F.mse_loss(preds_cont, targets_cont).item()
            total_mae += F.l1_loss(preds_cont, targets_cont).item()
            count += 1

        if method is not None:
            model.reset_hooks()

    return total_mse / count, total_mae / count


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    cfg = HookedTransformerConfig(
        n_layers=4, n_heads=4, d_model=256, d_head=64,
        d_vocab=4096, d_mlp=1024, n_ctx=1024, act_fn="gelu_new",
        normalization_type="LN", use_attn_result=True
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(loading.convert_gpt2_weights(hf_model, cfg), strict=False)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    train_loader, _, test_loader, data_min, data_max, data_mean, data_std = load_and_prepare_data()

    search_tokens = []
    for batch in train_loader:
        search_tokens.append(batch[0])
        if len(search_tokens) >= 4:
            break
    search_tokens = torch.cat(search_tokens, dim=0).to(device)
    n_search = min(200, search_tokens.size(0))

    tokens_orig = search_tokens[:n_search]
    tokens_shuf = search_tokens[:n_search].clone()
    perm = torch.randperm(tokens_shuf.size(1))
    tokens_shuf = tokens_shuf[:, perm]

    target_idx = SEQ_LEN - 1

    print("\n[1/3] LDA 搜索 Induction Heads...")
    top_head, all_heads, ref_cache = find_induction_heads(
        model, tokens_orig, tokens_shuf,
        cfg.n_layers, cfg.n_heads, target_idx, device
    )

    top_l, top_h = top_head

    model.eval()
    omega = 1.0 / EXPERIENCE_PERIOD

    print(f"\n[2/3] 在 Test 集合上评估各引导方式 (MSE / MAE)...")

    configs = [
        ("Normal ", None, None),
        (f"Cosine Bias L{top_l}H{top_h}", "cosine", top_head),
        ("Cosine Bias All Heads", "cosine", "all"),
        (f"Exp Decay Bias L{top_l}H{top_h} (sigma={SIGMA})", "exp_decay", top_head),
        (f"Exp Decay Bias All Heads (sigma={SIGMA})", "exp_decay", "all"),
        (f"Patching L{top_l}H{top_h}", "patch", top_head),
        ("Patching All Heads", "patch", "all"),
    ]

    test_results = {}
    for idx, (label, method, param) in tqdm(list(enumerate(configs)), desc="方法进度", total=len(configs)):
        mse, mae = run_eval(
            model, test_loader,
            data_min, data_max, data_mean, data_std,
            method, param, device, omega, ref_cache
        )
        test_results[idx] = (mse, mae)
        print(f"   {label} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    best_idx = min(test_results, key=lambda k: test_results[k][0])
    print(f"{'方法':<45} {'MSE':>8} {'MAE':>8}")

    for idx, (label, _, _) in enumerate(configs):
        mse, mae = test_results[idx]
        marker = " *" if idx == best_idx else ""
        print(f"{label:<45} {mse:>8.4f} {mae:>8.4f}{marker}")
    print("=" * 65)


if __name__ == "__main__":
    main()