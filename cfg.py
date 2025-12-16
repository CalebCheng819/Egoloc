# cfg.py
from __future__ import annotations
import os, yaml, argparse, json, time
from pathlib import Path
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dotset(cfg: Dict[str, Any], key: str, value: Any):
    """用 k1.k2.k3=val 覆盖嵌套字典。"""
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    # 尝试把字符串转成数值/布尔
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                if "." in value: value = float(value)
                else: value = int(value)
            except: pass
    cur[parts[-1]] = value

def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    for x in overrides:
        if "=" not in x:
            continue
        k, v = x.split("=", 1)
        dotset(cfg, k.strip(), v.strip())
    return cfg

def resolve_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """将像 '${ENV_NAME}' 的字符串替换为 os.environ 的值（若存在）。"""
    def _resolve(x):
        if isinstance(x, dict):
            return {k: _resolve(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_resolve(v) for v in x]
        if isinstance(x, str) and x.startswith("${") and x.endswith("}"):
            return os.environ.get(x[2:-1], "")
        return x
    return _resolve(cfg)

def save_config_snapshot(cfg: Dict[str, Any], save_dir: str, run_name: str):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(save_dir) / f"{run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[cfg] saved config snapshot to: {out_dir}/resolved_config.json")
    return out_dir

def print_config(cfg: Dict[str, Any]):
    print("="*60)
    print("[CONFIG] resolved training/inf config")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("="*60)

def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--config", type=str, default="config.yaml", help="path to config.yaml")
    ap.add_argument("--override", type=str, nargs="*", default=[], help="dotlist overrides like a.b.c=1 x.y=z")
    return ap

def load_config_from_cli():
    ap = add_common_args(argparse.ArgumentParser())
    # 你已有的参数也可以继续加，比如 --credentials/--video_type/--grid 等，但推荐都放 yaml
    args, _ = ap.parse_known_args()
    cfg = load_yaml(args.config)
    cfg = resolve_env(cfg)
    cfg = apply_overrides(cfg, args.override)
    return cfg, args
