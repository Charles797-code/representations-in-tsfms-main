import os
import subprocess
import sys
import torch

def main():
    

    config_files = [
        "random_univariate.yaml"
    ]


    for config in config_files:
        config_path = os.path.join("tsfm_similarity", "experiments", "similarity", "config", config)

        if not os.path.exists(config_path):
            print(f" 配置文件不存在：{config_path}")
            continue

        print(f"\nRunning experiment with config: {config}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "tsfm_similarity.experiments.similarity.similarity_experiment", "--config",
                 config_path],
                check=True,  # 命令执行失败时抛出异常
                capture_output=False,  # 实时输出实验日志
                text=True  # 输出为文本格式
            )

            print(f"Finished experiment with config: {config}")
            print("----------------------------------------")

        except subprocess.CalledProcessError as e:
            print(f"\n 执行配置 {config} 时出错：{e.stderr}")
            print(f"Finished experiment with config: {config} (failed)")
            print("----------------------------------------")
            continue

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
