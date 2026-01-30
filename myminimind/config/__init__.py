"""
配置包：训练参数的定义与加载。

  - TrainConfig：配置的「形状」和默认值，一般不用手写实例，用 get_config() 得到。
  - get_config()：入口函数。从默认 + .env/env + 配置文件 + 命令行拼出最终配置，返回一个 TrainConfig。

用法：
  from myminimind.config import get_config
  cfg = get_config()
  # 然后 cfg.batch_size、cfg.learning_rate、cfg.to_lm_config_kwargs() 等
"""

from .schema import TrainConfig
from .load import get_config

__all__ = ["TrainConfig", "get_config"]
