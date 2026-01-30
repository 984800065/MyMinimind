"""
配置加载：按「默认 → 配置文件 → 命令行」三层覆盖，得到最终的 TrainConfig。

为什么要分层：
  - 默认值：代码里写死一套合理默认。
  - 配置文件：不同实验用不同 json/yaml，不改代码。
  - 命令行：临时覆盖某几项（如 --batch-size 64），不用改文件。

get_config() 是唯一入口，训练脚本里调用 cfg = get_config() 即可。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import TrainConfig


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    """根据后缀把配置文件读成字典，只支持 .json / .yaml / .yml。"""
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("YAML 支持需要: pip install pyyaml") from None
    raise ValueError(f"不支持的配置文件格式: {suffix}")


def _bool_opt(s: str | None) -> bool | None:
    """把命令行传来的 0/1、true/false、yes 等转成 bool，None 保持 None（表示未传）。"""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("1", "true", "yes")


def _build_parser() -> argparse.ArgumentParser:
    """
    构建命令行解析器。所有参数 default=None，表示「没传就不覆盖」：
    这样在 get_config() 里只把「用户真正传了」的项覆盖到配置上，没传的用默认或配置文件里的值。
    """
    p = argparse.ArgumentParser(description="MiniMind 预训练")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    # nargs="?" + const="1"：只写 --use-moe 时当作 "1"（True），写 --use-moe 0 时为 "0"（False）
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")

    # 恢复与续训
    p.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # 实验与工具
    p.add_argument("--use-wandb", nargs="?", const="1", default=None, dest="use_wandb", help="启用 wandb")
    p.add_argument("--wandb-project", type=str, default=None, dest="wandb_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    return p


def get_config(args: list[str] | None = None) -> TrainConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 TrainConfig 实例。

    参数：
      args：通常不传，表示用 sys.argv（即当前进程的命令行）；传了则用这份列表解析，便于测试或二次封装。

    步骤简述：
      1. 用 TrainConfig() 无参构造：会用到 schema 里的默认值，并自动读 .env 和 TRAIN_* 环境变量，得到第一版字典。
      2. 若命令行带了 --config 且文件存在：用该文件内容覆盖字典里同名字段（只覆盖文件中出现的、且值非 None 的）。
      3. 若命令行带了其它参数（如 --batch-size 64）：再覆盖字典里对应字段；没带的保持上一步的值。
      4. 用最终字典构造 TrainConfig 并返回。
    """
    parser = _build_parser()
    parsed = parser.parse_args(args)

    # 第一层：默认值 + .env + 环境变量（TRAIN_*）。无参构造时 BaseSettings 会自动读 env
    config_dict = TrainConfig().model_dump()

    # 第二层：配置文件。只覆盖 config_dict 里已有的 key，且文件里值非 None 才覆盖
    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    # 第三层：命令行。只处理用户真的传了的参数（getattr 得到 None 表示没传，不覆盖）
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        # 命令行里布尔类参数可能是字符串 "0"/"1"，转成 bool
        if key in ("use_moe", "from_resume", "use_wandb", "use_compile"):
            val = _bool_opt(val)
        config_dict[key] = val

    return TrainConfig(**config_dict)
