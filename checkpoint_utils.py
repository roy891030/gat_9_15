import os
from typing import Any, Mapping

import torch

from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM, GATRegressor


CHECKPOINT_FORMAT = "explicit_v1"
CHECKPOINT_FORMAT_VERSION = 1


def _clone_state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def infer_legacy_model_type(state_dict: Mapping[str, Any]) -> str:
    keys = set(state_dict.keys())

    if any(
        key.startswith(prefix)
        for key in keys
        for prefix in ("gat_universe.", "factor_attention.", "attn_factor_head.", "factor_decoder.")
    ):
        return "dmfm"

    if any(
        key.startswith(prefix)
        for key in keys
        for prefix in ("gat1.", "gat2.", "norm.", "lin.")
    ):
        return "gat"

    if any(
        key.startswith(prefix)
        for key in keys
        for prefix in ("lstm.", "fc.")
    ):
        return "baseline_lstm"

    return "unknown"


def save_torch_checkpoint(
    path: str,
    model_type: str,
    model_or_state_dict: Any,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = (
        model_or_state_dict.state_dict()
        if hasattr(model_or_state_dict, "state_dict")
        else model_or_state_dict
    )
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": model_type,
        "config": dict(config or {}),
        "metadata": dict(metadata or {}),
        "state_dict": _clone_state_dict_to_cpu(state_dict),
    }
    torch.save(payload, path)
    return payload


def load_torch_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    raw = torch.load(path, map_location=map_location)

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], Mapping):
        payload = dict(raw)
        payload.setdefault("checkpoint_format", CHECKPOINT_FORMAT)
        payload.setdefault("format_version", CHECKPOINT_FORMAT_VERSION)
        payload.setdefault("config", {})
        payload.setdefault("metadata", {})
        payload["model_type"] = payload.get("model_type") or infer_legacy_model_type(payload["state_dict"])
        payload["is_legacy"] = False
        return payload

    if isinstance(raw, Mapping):
        return {
            "checkpoint_format": "legacy_state_dict",
            "format_version": 0,
            "model_type": infer_legacy_model_type(raw),
            "config": {},
            "metadata": {},
            "state_dict": dict(raw),
            "is_legacy": True,
        }

    raise TypeError(f"Unsupported checkpoint format in {path}")


def detect_checkpoint_model_type(path: str, map_location: str | torch.device = "cpu") -> str:
    payload = load_torch_checkpoint(path, map_location=map_location)
    return payload["model_type"]


def build_graph_model_from_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
    fallback_in_dim: int | None = None,
    fallback_hid: int = 64,
    fallback_heads: int = 2,
    fallback_dropout: float = 0.1,
    fallback_tanh_cap: float | None = 0.2,
) -> tuple[torch.nn.Module, dict[str, Any], list[str], list[str]]:
    payload = load_torch_checkpoint(path, map_location=map_location)
    model_type = payload["model_type"]
    config = payload.get("config", {})
    in_dim = int(config.get("in_dim", config.get("num_features", fallback_in_dim)))

    if model_type == "dmfm":
        model = DMFM(
            in_dim=in_dim,
            hidden_dim=int(config.get("hidden_dim", fallback_hid)),
            heads=int(config.get("heads", fallback_heads)),
            dropout=float(config.get("dropout", fallback_dropout)),
            use_factor_attention=bool(config.get("use_factor_attention", True)),
        )
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        return model, payload, missing, unexpected

    if model_type == "gat":
        tanh_cap = config["tanh_cap"] if "tanh_cap" in config else fallback_tanh_cap
        model = GATRegressor(
            in_dim=in_dim,
            hid=int(config.get("hid", fallback_hid)),
            heads=int(config.get("heads", fallback_heads)),
            dropout=float(config.get("dropout", fallback_dropout)),
            tanh_cap=tanh_cap,
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return model, payload, [], []

    raise ValueError(f"Unsupported graph checkpoint model_type={model_type!r} from {path}")


def describe_checkpoint(payload: Mapping[str, Any]) -> str:
    model_type = payload.get("model_type", "unknown")
    version = payload.get("format_version", 0)
    legacy = payload.get("is_legacy", False)
    return f"model_type={model_type} format_version={version} legacy={legacy}"

