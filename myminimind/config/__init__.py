"""
配置包：训练与推理参数的定义与加载。

  - PretrainConfig / get_pretrain_config()：预训练配置。
  - SFTConfig / get_sft_config()：Full SFT 配置。
  - DPOConfig / get_dpo_config()：DPO 配置。
  - InferConfig / get_infer_config()：推理/对话配置。

用法：
  from myminimind.config import get_pretrain_config, get_sft_config, get_dpo_config, get_infer_config
  cfg = get_pretrain_config()   # 预训练
  cfg = get_sft_config()       # Full SFT
  cfg = get_dpo_config()       # DPO
  infer_cfg = get_infer_config()  # 推理
"""

from .schema import DPOConfig, InferConfig, PretrainConfig, SFTConfig
from .load import get_dpo_config, get_pretrain_config, get_infer_config, get_sft_config

__all__ = ["DPOConfig", "InferConfig", "PretrainConfig", "SFTConfig", "get_dpo_config", "get_pretrain_config", "get_sft_config", "get_infer_config"]
