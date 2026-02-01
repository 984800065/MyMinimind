"""
配置包：训练与推理参数的定义与加载。

  - TrainConfig / get_config()：预训练配置。
  - InferConfig / get_infer_config()：推理/对话配置。

用法：
  from myminimind.config import get_config, get_infer_config
  cfg = get_config()       # 训练
  infer_cfg = get_infer_config()  # 推理
"""

from .schema import InferConfig, TrainConfig
from .load import get_config, get_infer_config

__all__ = ["InferConfig", "TrainConfig", "get_config", "get_infer_config"]
