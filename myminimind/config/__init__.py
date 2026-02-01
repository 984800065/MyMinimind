"""
配置包：训练与推理参数的定义与加载。

  - PretrainConfig / get_pretrain_config()：预训练配置。
  - SFTConfig / get_sft_config()：Full SFT 配置。
  - InferConfig / get_infer_config()：推理/对话配置。

用法：
  from myminimind.config import get_pretrain_config, get_sft_config, get_infer_config
  cfg = get_pretrain_config()   # 预训练
  cfg = get_sft_config()       # Full SFT
  infer_cfg = get_infer_config()  # 推理
"""

from .schema import InferConfig, PretrainConfig, SFTConfig
from .load import get_pretrain_config, get_infer_config, get_sft_config

__all__ = ["InferConfig", "PretrainConfig", "SFTConfig", "get_pretrain_config", "get_sft_config", "get_infer_config"]
