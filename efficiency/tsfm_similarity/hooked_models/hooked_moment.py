from typing import Any, Dict, List
from momentfm import MOMENTPipeline
from .hooked_ts_foundation_model import HookedTSFoundationModel
from .hook import Hook
import torch.nn as nn
import torch
import logging
import warnings
import os
import yaml
import json
from huggingface_hub import snapshot_download

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.utils._pytree._register_pytree_node.*"
)


class HookedMOMENT(HookedTSFoundationModel):
    """
    Hooked MOMENT model, from Auton Lab, Carnegie Mellon University.
    input shape: (B, C, L)
    Works with multivariate data, C is the number of channels.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)
        # 保存项目配置到实例，避免后续缺失
        self.config = config

    def setup_model(self, config: Dict[str, Any]) -> nn.Module:

        """
        修复核心：显式下载并加载 MOMENT 模型原生 config，传入初始化
        """
        model_version = config["model_version"]
        logging.info(f"Loading MOMENT variant: {model_version}")

        # 步骤1：下载模型快照（包含原生 config 文件）到本地缓存
        try:
            model_cache_dir = snapshot_download(repo_id=model_version)
            logging.info(f"Model cached at: {model_cache_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to download MOMENT model: {e}")

        # 步骤2：加载 MOMENT 模型原生 config 文件（兼容 yaml/json）
        config_paths = [
            os.path.join(model_cache_dir, "config.yaml"),
            os.path.join(model_cache_dir, "config.json"),
            os.path.join(model_cache_dir, "model_config.yaml")
        ]
        moment_config = None
        for cfg_path in config_paths:
            if os.path.exists(cfg_path):
                logging.info(f"Loading model config from: {cfg_path}")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    if cfg_path.endswith(".yaml") or cfg_path.endswith(".yml"):
                        moment_config = yaml.safe_load(f)
                    else:
                        moment_config = json.load(f)
                break
        if moment_config is None:
            raise FileNotFoundError(f"Could not find MOMENT config file in {model_cache_dir}")

        # 步骤3：正确初始化 MOMENTPipeline（传入原生 config）
        try:
            # 先加载预训练权重，再传入原生 config 初始化
            moment = MOMENTPipeline(
                config=moment_config,  # 核心：传入模型原生 config
                pretrained_model_name_or_path=model_version
            )
            # 初始化模型权重
            moment.init()
            logging.info("MOMENTPipeline initialized with native config")
        except Exception as e:
            raise TypeError(f"Failed to init MOMENTPipeline: {e}")

        # 可选优化：移到 GPU（和 Chronos 模型保持一致）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            moment.model = moment.model.to(device)
            logging.info(f"MOMENT model moved to GPU (cuda:0)")

        logging.info(f"Successfully loaded MOMENT variant: {model_version}")
        return moment

    def setup_hooks(
        self, backward=False, set_encoder=True, set_ln=False
    ) -> Dict[str, Hook]:
        encoder_hooks = []
        layer_norms = []
        for block in self.model.encoder.block:
            if set_ln:
                hook_ln = Hook(block.layer[0].layer_norm, backward=backward)
                layer_norms.append(hook_ln)
            if set_encoder:
                hook_block = Hook(block, backward=backward)
                encoder_hooks.append(hook_block)
        hooks = {
            "encoder": encoder_hooks,
            "layer_norms": layer_norms,
        }
        return hooks

    def forward(self, x_BCL: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.extract_time_series_embeddings(x_BCL, *args, **kwargs)

    def process_input(self, x_BCL: torch.Tensor) -> torch.Tensor:
        if len(x_BCL.shape) != 3:
            raise ValueError("Bad input shape, expected (B, C, L)")
        return x_BCL

    def extract_time_series_embeddings(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Extract time series embeddings from the MOMENT.
        ## inference function, processing the input
        Args:
            x: Input time series data, with shape (B, C, L).

        Returns:
            Time series embeddings, with shape (B, D) - each of B time series is represented by a single embedding of size D.
        """
        x_BCL = self.process_input(x_BCL)
        # 确保输入数据和模型同设备
        if hasattr(self.model, "model") and next(self.model.model.parameters()).device != x_BCL.device:
            x_BCL = x_BCL.to(next(self.model.model.parameters()).device)
        embeddings_BD = self.model.embed(x_enc=x_BCL).embeddings
        return embeddings_BD

    def get_encoder_representations(
        self, x_BCL: torch.Tensor, normalized_by_final_ln: bool = True, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Get the encoder representation of the input. Specific for models with encoder.

        Args:
            x: Input time series data, with shape (B, C, L).
            normalized_by_final_ln: Whether to normalize the output by the final layer norm.

        Returns:
            List of encoder representations, with shape (B, P, D)
        Recommended to average along P axis to get a single embedding for each time series.
        """
        batch_size, num_channels, seq_len = x_BCL.shape
        self.forward(x_BCL, *args, **kwargs)
        representations = []
        hook_list = self.hooks["encoder"]
        for hook in hook_list:
            level_representations_BaCPD = hook.output
            level_representations_BaCPD = self.model.encoder.final_layer_norm(
                level_representations_BaCPD
            )
            # BaC means single axis of size B * C
            # leave vectors in shape B, P, D, collapse C with mean
            level_representations_BaPD = level_representations_BaCPD.view(
                batch_size, num_channels, -1, level_representations_BaCPD.shape[-1]
            )
            if normalized_by_final_ln:
                level_representations_BaPD = self.model.encoder.final_layer_norm(
                    level_representations_BaPD
                )
            level_representations_BPD = level_representations_BaPD.mean(dim=1)
            representations.append(level_representations_BPD)
        return representations

    def get_layer_norms_representations(
        self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Get the layer norm representation of the input. Specific for models with layer norms.

        Args:
            x: Input time series data, with shape (B, C, L).

        Returns:
            List of layer norm representations, with shape (B, L, D)
        Recommended to average along L axis to get a single embedding for each time series.
        """
        batch_size, num_channels, seq_len = x_BCL.shape
        self.forward(x_BCL, *args, **kwargs)
        representations = []
        hook_list = self.hooks["layer_norms"]
        for hook in hook_list:
            level_representations_BaCPD = hook.output
            # BaC means single axis of size B * C
            # leave vectors in shape B, P, D, collapse C with mean
            level_representations_BaPD = level_representations_BaCPD.view(
                batch_size, num_channels, -1, level_representations_BaCPD.shape[-1]
            )
            level_representations_BPD = level_representations_BaPD.mean(dim=1)
            representations.append(level_representations_BPD)
        return representations