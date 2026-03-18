import torch
import pandas as pd
import os

from efficiency.tsfm_similarity.datautils.data_generator import DataGenerator


def main():
    print("初始化数据生成器...")
    # 1. 实例化生成器
    generator = DataGenerator(random_seed=42)

    # ==========================================
    # 选项 A: 生成正弦波合成数据 (Synthetic Data)
    # ==========================================
    print("正在生成合成正弦波数据...")
    y_syn, c_syn = generator.generate_synthetic_data(
        n_samples=200,  # 生成 200 条数据
        seq_len=512,  # 序列长度与你的 Chronos/MOMENT 上下文对齐
        n_channels=1,  # 单变量时间序列
        noise_std=0.1  # 噪音强度
    )

    # y_syn 的形状为 (200, 1, 512)，去除通道维度以便保存
    y_syn_np = y_syn.squeeze(1).numpy()
    c_syn_np = c_syn.numpy()

    # 封装为包含 "series" 列的 DataFrame
    df_syn = pd.DataFrame({
        "series": [y_syn_np[i] for i in range(y_syn_np.shape[0])],
        "label": c_syn_np  # 保存对应的类别/频率标签
    })

    os.makedirs("datasets", exist_ok=True)
    syn_save_path = "datasets/my_synthetic_data.parquet"
    df_syn.to_parquet(syn_save_path, index=False)
    print(f"✅ 合成正弦波数据已保存至: {syn_save_path}")

    # ==========================================
    # 选项 B: 生成纯随机分布数据 (Random Data)
    # ==========================================
    print("正在生成随机高斯分布数据...")
    y_rand = generator.generate_random_data(
        n_samples=200,
        seq_len=512,
        n_channels=1,
        distribution="normal",
        mean=0,
        std=1
    )

    y_rand_np = y_rand.squeeze(1).numpy()
    df_rand = pd.DataFrame({
        "series": [y_rand_np[i] for i in range(y_rand_np.shape[0])]
    })

    rand_save_path = "datasets/my_random_data.parquet"
    df_rand.to_parquet(rand_save_path, index=False)
    print(f"✅ 随机高斯数据已保存至: {rand_save_path}")


if __name__ == "__main__":
    main()