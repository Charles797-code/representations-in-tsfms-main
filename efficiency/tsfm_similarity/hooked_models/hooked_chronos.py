from typing import Any, Dict, List
from .hooked_ts_foundation_model import HookedTSFoundationModel
from chronos import ChronosPipeline
from .hook import Hook
import torch.nn as nn
import torch
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.utils._pytree._register_pytree_node.*"
)


class HookedChronos(HookedTSFoundationModel):
    """
    Hooked Chronos model, from Amazon Science.
    input shape: (B, C, L)
    WARNING - works only with univariate data, so C has to be 1.
    C is always collapsed in the process_input method.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        super().__init__(config=config)

    def setup_model(self, config: Dict[str, Any]) -> nn.Module:
        model_version = config["model_version"]

        # 优化项：检测当前环境并自动使用 device_map，这是最安全的挂载方式
        device_string = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {
            "device_map": device_string,
        }

        # 大模型启用 fp16 以节省显存
        if "large" in model_version.lower():
            model_kwargs["torch_dtype"] = torch.float16
            logging.warning(f"Detected large model {model_version}, using float16 to save VRAM")

        # 加载 Pipeline，HuggingFace 会自动把底层模型挂载到 GPU
        chronos = ChronosPipeline.from_pretrained(model_version, **model_kwargs)

        logging.info(f"Loaded Chronos variant {model_version} on device: {device_string}")
        return chronos

    def setup_hooks(self, backward=False) -> Dict[str, Hook]:
        encoder_hooks = []
        for block in self.model.model.model.encoder.block:
            hook = Hook(block, backward=backward)
            encoder_hooks.append(hook)
        final_layer_norm = Hook(
            self.model.model.model.encoder.final_layer_norm, backward=backward
        )
        encoder_hooks.append(final_layer_norm)
        return {"encoder": encoder_hooks}

    def forward(self, x_BCL: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.extract_time_series_embeddings(x_BCL, *args, **kwargs)

    def process_input(self, x_BCL: torch.Tensor) -> torch.Tensor:
        if len(x_BCL.shape) != 3:
            raise ValueError("Bad input shape, expected (B, C, L)")
        if x_BCL.shape[1] != 1:
            raise ValueError(
                "Chronos model works only with univariate data, C has to be 1"
            )
        x_BL = x_BCL.squeeze(1)
        return x_BL

    def extract_time_series_embeddings(
            self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Extract time series embeddings from the Chronos.
        """
        x_BL = self.process_input(x_BCL)

        # === 核心修复点：撤销提前转移至 GPU 的操作 ===
        # 强制将输入保留在 CPU 并转为 float32，满足 Chronos Tokenizer 的内部运算要求。
        # Pipeline 内部会在分箱操作完成后，自动将其放入底层模型所在的 GPU。
        x_BL = x_BL.cpu().to(torch.float32)

        with torch.no_grad():
            embeddings, tokenizer_state = self.model.embed(x_BL)

        embeddings_BLD = embeddings
        return embeddings_BLD

    def get_encoder_representations(
            self, x_BCL: torch.Tensor, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Get the encoder representation of the input.
        """
        self.forward(x_BCL, *args, **kwargs)
        representations = []
        for hook in self.hooks["encoder"]:
            level_representations_BLD = hook.output
            representations.append(level_representations_BLD)
        return representations