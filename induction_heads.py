import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.loading_from_pretrained as loading
import matplotlib.pyplot as plt


def generate_sinusoidal_data(batch_size, seq_len, vocab_size):
    data = []
    for _ in range(batch_size):
        f = np.random.uniform(10, 50)
        a = np.random.uniform(0.5, 1.0)
        t = np.arange(seq_len)
        wave = a * np.sin(2 * np.pi * t / f)
        wave_norm = (wave + 1.0) / 2.0
        tokens = (wave_norm * (vocab_size - 1)).clip(0, vocab_size - 1).astype(int)
        data.append(tokens)
    return torch.tensor(np.array(data), dtype=torch.long)


def generate_constant_data(batch_size, seq_len, vocab_size):
    data = []
    for _ in range(batch_size):
        m = np.random.uniform(-0.02, 0.02)
        b = np.random.uniform(-0.5, 0.5)
        t = np.arange(seq_len)
        line = m * t + b
        line_norm = (line + 1.0) / 2.0
        tokens = (line_norm * (vocab_size - 1)).clip(0, vocab_size - 1).astype(int)
        data.append(tokens)
    return torch.tensor(np.array(data), dtype=torch.long)


def tokens_to_continuous_data(tokens, vocab_size):
    wave_norm = tokens.float() / (vocab_size - 1)
    return (wave_norm * 2.0) - 1.0


def main():
    model_path = r"D:\new_representations\representations-in-tsfms-main\output\custom-gpt2-4l4h\run-3\checkpoint-final"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    cfg = HookedTransformerConfig(
        n_layers=4, n_heads=4, d_model=256, d_head=64,
        d_vocab=4096, d_mlp=1024, n_ctx=1024, act_fn="gelu_new",
        normalization_type="LN",
        use_attn_result=True
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(loading.convert_gpt2_weights(hf_model, cfg), strict=False)
    model.to(device)

    vocab_size = 4096

    search_seq_len = 64
    sin_tokens = generate_sinusoidal_data(50, search_seq_len, vocab_size).to(device)
    const_tokens = generate_constant_data(50, search_seq_len, vocab_size).to(device)
    all_tokens = torch.cat([sin_tokens, const_tokens], dim=0)
    labels = np.array([1] * 50 + [0] * 50)

    _, search_cache = model.run_with_cache(all_tokens)
    target_token_idx = search_seq_len - 1

    global_best_ldr = -1.0
    global_best_layer = -1
    global_best_head = -1

    for l in range(1, cfg.n_layers):
        layer_results = search_cache["result", l]
        for h in range(cfg.n_heads):
            X = layer_results[:, target_token_idx, h, :].cpu().numpy()
            y = labels
            lda = LinearDiscriminantAnalysis(n_components=1)
            try:
                X_proj = lda.fit_transform(X, y).flatten()
                mu_s, mu_c = X_proj[y == 1].mean(), X_proj[y == 0].mean()
                var_s, var_c = X_proj[y == 1].var(), X_proj[y == 0].var()
                ldr_score = ((mu_s - mu_c) ** 2) / (var_s + var_c + 1e-8)
            except:
                ldr_score = 0.0

            if ldr_score > global_best_ldr:
                global_best_ldr = ldr_score
                global_best_layer = l
                global_best_head = h



    test_seq_len = 256

    true_period = 30.0

    t_test = np.arange(test_seq_len)
    wave_test = 0.8 * np.sin(2 * np.pi * t_test / true_period)
    wave_norm_test = (wave_test + 1.0) / 2.0
    tokens_test = (wave_norm_test * (vocab_size - 1)).clip(0, vocab_size - 1).astype(int)
    periodic_test = torch.tensor(np.array([tokens_test]), dtype=torch.long).to(device)

    targets_tokens = periodic_test[0, 1:]
    targets_cont = tokens_to_continuous_data(targets_tokens, vocab_size)


    empirical_omega = 1.0 / true_period
    empirical_lambda = 0.8
    empirical_phi = 0.8


    print("\n[3] 验证测试：对比 Normal 和 经验 Resonance")

    # 1. 正常推理 (Baseline)
    normal_logits = model(periodic_test)
    preds_normal_tokens = torch.argmax(normal_logits[0, :-1], dim=-1)
    preds_normal_cont = tokens_to_continuous_data(preds_normal_tokens, vocab_size)
    mse_normal = F.mse_loss(preds_normal_cont, targets_cont)
    mae_normal = F.l1_loss(preds_normal_cont, targets_cont)


    def empirical_resonance_hook(value, hook):
        q_len, k_len = value.shape[-2], value.shape[-1]
        q_idx = torch.arange(q_len, device=value.device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=value.device).unsqueeze(0)
        delta_t = torch.abs(q_idx - k_idx).float()


        bias = empirical_lambda * torch.cos(2 * math.pi * empirical_omega * delta_t + empirical_phi)


        new_value = value.clone()
        new_value[:, global_best_head, :, :] = new_value[:, global_best_head, :, :] + bias
        return new_value

    hook_attn_name = f"blocks.{global_best_layer}.attn.hook_attn_scores"
    model.add_hook(hook_attn_name, empirical_resonance_hook)


    resonant_logits = model(periodic_test)
    preds_resonant_tokens = torch.argmax(resonant_logits[0, :-1], dim=-1)
    preds_resonant_cont = tokens_to_continuous_data(preds_resonant_tokens, vocab_size)

    mse_resonant = F.mse_loss(preds_resonant_cont, targets_cont)
    mae_resonant = F.l1_loss(preds_resonant_cont, targets_cont)
    model.reset_hooks()


    print(f"\n正常推理:")
    print(f"   MSE: {mse_normal.item():.6f} | MAE: {mae_normal.item():.6f}")

    print(f"\nEmpirical Resonance:")
    print(f"   MSE: {mse_resonant.item():.6f} | MAE: {mae_resonant.item():.6f}")

    mse_decrease = (mse_normal.item() - mse_resonant.item()) / (mse_normal.item() + 1e-8) * 100


    plot_len = 100
    t_steps = np.arange(plot_len)
    y_true = targets_cont[-plot_len:].cpu().numpy()
    y_norm = preds_normal_cont[-plot_len:].cpu().numpy()
    y_res = preds_resonant_cont[-plot_len:].cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(t_steps, y_true, label="True Sinusoid Target", color="black", linewidth=2, linestyle="--")
    plt.plot(t_steps, y_norm, label=f"Normal Predict (MSE: {mse_normal.item():.2f})", color="blue", alpha=0.7)
    plt.plot(t_steps, y_res, label=f"Empirical Resonant Guided (MSE: {mse_resonant.item():.2f})", color="green",
             alpha=0.9)

    plt.title("Empirical Resonance Attention: Using True Physical Priors")
    plt.xlabel("Time Steps")
    plt.ylabel("Continuous Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()