import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.loading_from_pretrained as loading


def generate_discrete_periodic_data(seq_len, period, vocab_size, device):
    t = torch.arange(seq_len, dtype=torch.float32)
    wave = torch.sin(2 * math.pi * t / period)
    wave_norm = (wave + 1.0) / 2.0
    tokens = (wave_norm * (vocab_size - 1)).long()

    return tokens.unsqueeze(0).to(device)


def tokens_to_continuous_data(tokens, vocab_size):
    wave_norm = tokens.float() / (vocab_size - 1)
    wave_continuous = (wave_norm * 2.0) - 1.0
    return wave_continuous


def main():
    model_path = r"D:\new_representations\representations-in-tsfms-main\output\custom-gpt2-4l4h\run-3\checkpoint-final"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    cfg = HookedTransformerConfig(
        n_layers=4, n_heads=4, d_model=256, d_head=64,
        d_vocab=4096, d_mlp=1024, n_ctx=1024, act_fn="gelu_new", normalization_type="LN"
    )
    model = HookedTransformer(cfg)
    state_dict = loading.convert_gpt2_weights(hf_model, cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    seq_half_len = 64
    vocab_size = 4096

    random_half = torch.randint(10, vocab_size - 10, (1, seq_half_len)).to(device)
    test_tokens = torch.cat([random_half, random_half], dim=1)
    _, cache = model.run_with_cache(test_tokens)

    global_best_score = -float('inf')
    global_best_layer = -1
    global_best_head = -1

    best_prefix = 0.0
    best_copy = 0.0

    W_E = model.W_E
    W_U = model.W_U

    for l in range(1, cfg.n_layers):
        attn_pattern = cache["pattern", l][0]
        for h in range(cfg.n_heads):
            # 计算前缀匹配分数 (QK回路)
            prefix_score = attn_pattern[h].diagonal(-seq_half_len).mean().item()

            # 计算复制分数 (OV回路)
            W_V = model.W_V[l, h]
            W_O = model.W_O[l, h]
            OV_matrix = W_E @ W_V @ W_O @ W_U
            diag_mean = OV_matrix.diag().mean().item()
            total_mean = OV_matrix.mean().item()
            copy_score = diag_mean - total_mean

            # 计算总分
            total_score = prefix_score + copy_score

            if total_score > global_best_score:
                global_best_score = total_score
                global_best_layer = l
                global_best_head = h
                best_prefix = prefix_score
                best_copy = copy_score

    print(
        f"induct head: L{global_best_layer}H{global_best_head} | 综合得分: {global_best_score:.4f} (Prefix: {best_prefix:.4f}, Copy: {best_copy:.4f})")

    periodic_tokens = generate_discrete_periodic_data(seq_len=256, period=32, vocab_size=vocab_size, device=device)

    targets_tokens = periodic_tokens[0, 1:]
    targets_continuous = tokens_to_continuous_data(targets_tokens, vocab_size)

    normal_logits = model(periodic_tokens)


    preds_normal_tokens = torch.argmax(normal_logits[0, :-1], dim=-1)
    preds_normal_continuous = tokens_to_continuous_data(preds_normal_tokens, vocab_size)


    mse_normal = F.mse_loss(preds_normal_continuous, targets_continuous)
    mae_normal = F.l1_loss(preds_normal_continuous, targets_continuous)


    def ablate_head_hook(value, hook):
        value[:, :, global_best_head, :] = 0.0
        return value

    hook_name = f"blocks.{global_best_layer}.attn.hook_z"
    model.add_hook(hook_name, ablate_head_hook)

    ablated_logits = model(periodic_tokens)


    preds_ablated_tokens = torch.argmax(ablated_logits[0, :-1], dim=-1)
    preds_ablated_continuous = tokens_to_continuous_data(preds_ablated_tokens, vocab_size)


    mse_ablated = F.mse_loss(preds_ablated_continuous, targets_continuous)
    mae_ablated = F.l1_loss(preds_ablated_continuous, targets_continuous)

    model.reset_hooks()


    print("-" * 50)
    print(f"正常预测")
    print(f"MSE: {mse_normal.item():.6f} | MAE: {mae_normal.item():.6f}")

    print(f"\n置零 induct head ")
    print(f"MSE: {mse_ablated.item():.6f} | MAE: {mae_ablated.item():.6f}")

    mse_increase = (mse_ablated.item() - mse_normal.item()) / (mse_normal.item() + 1e-8) * 100
    mae_increase = (mae_ablated.item() - mae_normal.item()) / (mae_normal.item() + 1e-8) * 100

    print("-" * 50)
    print(f"MSE 误差上升比例: {mse_increase:.2f}%")
    print(f"MAE 误差上升比例: {mae_increase:.2f}%")


if __name__ == "__main__":
    main()