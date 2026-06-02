from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Dict, Iterator, List, Optional


@dataclass
class TimingNode:
    id: int
    key: str
    label: str
    elapsed: float = 0.0
    self_elapsed: float = 0.0
    count: int = 1
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    kind: str = "scope"


@dataclass
class _ActiveSpan:
    node_id: int
    start_time: float
    child_elapsed: float = 0.0


@dataclass
class TimingStats:
    durations: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    nodes: Dict[int, TimingNode] = field(default_factory=dict)
    root_ids: List[int] = field(default_factory=list)
    _stack: List[_ActiveSpan] = field(default_factory=list)
    _next_node_id: int = 1

    def add(self, key: str, elapsed: float) -> None:
        self.durations[key] = self.durations.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1
        self._record_measurement(key=key, elapsed=elapsed, label=key)

    def get_duration(self, key: str) -> float:
        return self.durations.get(key, 0.0)

    def get_count(self, key: str) -> int:
        return self.counts.get(key, 0)

    @contextmanager
    def scope(self, key: str, label: Optional[str] = None) -> Iterator[None]:
        node_id = self._next_node_id
        self._next_node_id += 1

        parent_id = self._stack[-1].node_id if self._stack else None
        node = TimingNode(
            id=node_id,
            key=key,
            label=label or key,
            parent_id=parent_id,
        )
        self.nodes[node_id] = node

        if parent_id is None:
            self.root_ids.append(node_id)
        else:
            self.nodes[parent_id].children.append(node_id)

        active = _ActiveSpan(node_id=node_id, start_time=time.perf_counter())
        self._stack.append(active)
        try:
            yield
        finally:
            finished = self._stack.pop()
            elapsed = time.perf_counter() - finished.start_time
            node.elapsed = elapsed
            node.self_elapsed = max(0.0, elapsed - finished.child_elapsed)
            self.durations[key] = self.durations.get(key, 0.0) + elapsed
            self.counts[key] = self.counts.get(key, 0) + 1
            if self._stack:
                self._stack[-1].child_elapsed += elapsed

    def report_lines(
        self,
        total_elapsed: float,
        top_n_groups: int = 15,
        top_n_self: int = 12,
    ) -> List[str]:
        lines = [
            "Detailed timing:",
            "By scope:",
        ]

        for root_id in self.root_ids:
            self._append_tree_lines(lines, root_id=root_id, indent=0)

        grouped = self._group_scope_stats()
        if grouped:
            lines.append("Top aggregated scopes:")
            for key, stats in sorted(
                grouped.items(),
                key=lambda item: item[1]["total"],
                reverse=True,
            )[:top_n_groups]:
                total = float(stats["total"])
                count = int(stats["count"])
                self_total = float(stats["self"])
                share = (total / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
                avg = total / count if count else 0.0
                lines.append(
                    f"- {key}: total={total:.3f}s, self={self_total:.3f}s, "
                    f"avg={avg:.3f}s, count={count}, share={share:.1f}%"
                )

        self_heavy_nodes = [
            node for node in self.nodes.values() if node.kind == "scope" and node.self_elapsed > 0
        ]
        if self_heavy_nodes:
            lines.append("Top self-time scopes:")
            for node in sorted(
                self_heavy_nodes,
                key=lambda item: item.self_elapsed,
                reverse=True,
            )[:top_n_self]:
                lines.append(
                    f"- {node.label}: self={node.self_elapsed:.3f}s, total={node.elapsed:.3f}s"
                )

        return lines

    def _record_measurement(self, key: str, elapsed: float, label: str) -> None:
        node_id = self._next_node_id
        self._next_node_id += 1

        parent_id = self._stack[-1].node_id if self._stack else None
        node = TimingNode(
            id=node_id,
            key=key,
            label=label,
            elapsed=elapsed,
            self_elapsed=elapsed,
            parent_id=parent_id,
            kind="measurement",
        )
        self.nodes[node_id] = node

        if parent_id is None:
            self.root_ids.append(node_id)
        else:
            self.nodes[parent_id].children.append(node_id)
            self._stack[-1].child_elapsed += elapsed

    def _append_tree_lines(self, lines: List[str], root_id: int, indent: int) -> None:
        node = self.nodes[root_id]
        prefix = "  " * indent
        detail = f"{prefix}- {node.label}: {node.elapsed:.3f}s"
        if node.kind == "scope":
            detail += f" (self={node.self_elapsed:.3f}s)"
        lines.append(detail)
        for child_id in node.children:
            self._append_tree_lines(lines, root_id=child_id, indent=indent + 1)

    def _group_scope_stats(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, Dict[str, float]] = {}
        for node in self.nodes.values():
            if node.kind != "scope":
                continue
            current = grouped.setdefault(node.key, {"total": 0.0, "self": 0.0, "count": 0.0})
            current["total"] += node.elapsed
            current["self"] += node.self_elapsed
            current["count"] += node.count
        return grouped


_ACTIVE_TIMING: Optional[TimingStats] = None


def get_active_timing() -> Optional[TimingStats]:
    return _ACTIVE_TIMING


def set_active_timing(timing: Optional[TimingStats]) -> None:
    global _ACTIVE_TIMING
    _ACTIVE_TIMING = timing
