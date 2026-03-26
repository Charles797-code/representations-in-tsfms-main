import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


def generate_sinusoidal_tokens(seq_len, period=30.0, amplitude=0.8, vocab_size=50257):
    t = np.arange(seq_len)
    wave = amplitude * np.sin(2 * np.pi * t / period)
    wave_norm = (wave + 1.0) / 2.0
    tokens = (wave_norm * (vocab_size - 1)).clip(0, vocab_size - 1).astype(int)
    return torch.tensor([tokens], dtype=torch.long)


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "gpt2"
    SEQ_LEN = 128

    print(f"加载模型: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    tokens = generate_sinusoidal_tokens(SEQ_LEN).to(DEVICE)

    _, cache = model.run_with_cache(tokens, names_filter=lambda n: "pattern" in n)

    rows = n_layers
    cols = n_heads

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2 + 1, rows * 2 + 1), squeeze=False)
    fig.suptitle(f"GPT-2 ({MODEL_NAME}) — Attention Heatmaps | Input: Sinusoidal", fontsize=14)

    vmax = 0.5

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h]
            attn = cache["pattern", l][0, h].cpu().numpy()
            ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"L{l}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
