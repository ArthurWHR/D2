"""Shared configuration loader.

All numeric settings live in ``config.py``, which is not tracked by git.
Copy ``config.example.py`` to ``config.py`` and fill in your own values.
"""

import sys

# 统一使用 UTF-8 输出，避免 Windows GBK 控制台无法编码 emoji/中文
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

try:
    import config as cfg
except ImportError:
    sys.exit(
        "config.py not found. Copy config.example.py to config.py and fill in the values.\n"
        "未找到 config.py：请将 config.example.py 复制为 config.py 并填入取值。"
    )


def require(name, value):
    """Return ``value``, or exit with a clear message when it is not configured."""
    if value is None:
        sys.exit(f"config.py: '{name}' is not set. / 配置项 '{name}' 未填写。")
    return value
