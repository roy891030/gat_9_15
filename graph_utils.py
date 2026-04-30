# -*- coding: utf-8 -*-
"""Utilities for dynamic weighted graph artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


DYNAMIC_GRAPH_META = "dynamic_graph_meta.json"
INDUSTRY_EDGE_INDICES = "dynamic_industry_edge_indices.pt"
INDUSTRY_EDGE_ATTRS = "dynamic_industry_edge_attrs.pt"
UNIVERSE_EDGE_INDICES = "dynamic_universe_edge_indices.pt"
UNIVERSE_EDGE_ATTRS = "dynamic_universe_edge_attrs.pt"


@dataclass
class DynamicGraphs:
    industry_edge_indices: list[torch.Tensor]
    industry_edge_attrs: list[torch.Tensor]
    universe_edge_indices: list[torch.Tensor]
    universe_edge_attrs: list[torch.Tensor]
    edge_dim: int

    @property
    def available(self) -> bool:
        return (
            bool(self.industry_edge_indices)
            and bool(self.industry_edge_attrs)
            and bool(self.universe_edge_indices)
            and bool(self.universe_edge_attrs)
        )


def dynamic_graph_files_exist(artifact_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(artifact_dir, name))
        for name in (
            INDUSTRY_EDGE_INDICES,
            INDUSTRY_EDGE_ATTRS,
            UNIVERSE_EDGE_INDICES,
            UNIVERSE_EDGE_ATTRS,
        )
    )


def load_dynamic_graphs(artifact_dir: str, map_location: str | torch.device = "cpu") -> Optional[DynamicGraphs]:
    if not dynamic_graph_files_exist(artifact_dir):
        return None

    industry_edge_indices = torch.load(
        os.path.join(artifact_dir, INDUSTRY_EDGE_INDICES),
        map_location=map_location,
    )
    industry_edge_attrs = torch.load(
        os.path.join(artifact_dir, INDUSTRY_EDGE_ATTRS),
        map_location=map_location,
    )
    universe_edge_indices = torch.load(
        os.path.join(artifact_dir, UNIVERSE_EDGE_INDICES),
        map_location=map_location,
    )
    universe_edge_attrs = torch.load(
        os.path.join(artifact_dir, UNIVERSE_EDGE_ATTRS),
        map_location=map_location,
    )

    if not industry_edge_attrs or not universe_edge_attrs:
        return None
    edge_dim = int(industry_edge_attrs[0].shape[-1])
    return DynamicGraphs(
        industry_edge_indices=industry_edge_indices,
        industry_edge_attrs=industry_edge_attrs,
        universe_edge_indices=universe_edge_indices,
        universe_edge_attrs=universe_edge_attrs,
        edge_dim=edge_dim,
    )


def filter_edge_index_and_attr(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], mask: torch.Tensor):
    """Filter full-universe edge indices to a masked node set and remap node ids."""
    old_to_new = torch.full((mask.size(0),), -1, dtype=torch.long, device=mask.device)
    old_to_new[mask] = torch.arange(mask.sum(), device=mask.device)

    src, dst = edge_index[0], edge_index[1]
    valid_edges = mask[src] & mask[dst]
    if valid_edges.sum() == 0:
        empty_edge = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_attr = (
            torch.empty((0, edge_attr.shape[-1]), dtype=edge_attr.dtype, device=edge_attr.device)
            if edge_attr is not None
            else None
        )
        return empty_edge, empty_attr

    filtered_edge_index = torch.stack(
        [old_to_new[src[valid_edges]], old_to_new[dst[valid_edges]]],
        dim=0,
    )
    filtered_edge_attr = edge_attr[valid_edges] if edge_attr is not None else None
    return filtered_edge_index, filtered_edge_attr


def graph_at(
    dynamic_graphs: Optional[DynamicGraphs],
    graph_name: str,
    t: int,
    fallback_edge_index: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    device: str | torch.device | None = None,
):
    """Return edge_index and edge_attr for one timestamp, with static fallback."""
    if dynamic_graphs is None:
        edge_index = fallback_edge_index
        edge_attr = None
    elif graph_name == "industry":
        edge_index = dynamic_graphs.industry_edge_indices[t]
        edge_attr = dynamic_graphs.industry_edge_attrs[t]
    elif graph_name == "universe":
        edge_index = dynamic_graphs.universe_edge_indices[t]
        edge_attr = dynamic_graphs.universe_edge_attrs[t]
    else:
        raise ValueError(f"Unknown graph_name={graph_name!r}")

    if device is not None:
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

    if mask is not None:
        edge_index, edge_attr = filter_edge_index_and_attr(edge_index, edge_attr, mask)

    return edge_index.long(), edge_attr
