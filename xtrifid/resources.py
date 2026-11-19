"""Read Linux headroom before allocating meshes or loading a local model."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30


@dataclass(frozen=True)
class Capacity:
    headroom: int  # bytes available after the desktop and cgroup reserves
    cpus: int

    @property
    def threads(self):
        return min(6, max(1, self.cpus // 2))


def _groups(proc, pid):
    membership = (proc / str(pid or "self") / "cgroup").read_text().splitlines()
    group = next((line[3:] for line in membership if line.startswith("0::")), "")
    if not group.startswith("/") or ".." in Path(group).parts:
        raise ValueError("unknown cgroup")
    for line in (proc / "self/mountinfo").read_text().splitlines():
        if " - cgroup2 " not in line:
            continue
        # mountinfo encodes spaces and backslashes using octal escapes.
        root, mount = (
            Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field))
            for field in line.split()[3:5]
        )
        member = Path(group)
        relative = (
            member.relative_to(root) if member.is_relative_to(root) else member.relative_to("/")
        )
        leaf = mount / relative
        if leaf.is_dir():
            while leaf != mount:
                yield leaf
                leaf = leaf.parent
            yield mount
            return
    raise ValueError("unreadable cgroup hierarchy")


def _value(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None  # root groups and disabled controllers omit their limit files


def _inactive_cache(group):
    data = _value(group / "memory.stat")
    if data is None:
        return 0
    try:
        stats = dict(line.split() for line in data.splitlines())
        cold, *unsafe = (
            int(stats[name])
            for name in (
                "inactive_file",
                "file_mapped",
                "shmem",
                "file_dirty",
                "file_writeback",
            )
        )
        if cold >= 0 and all(value >= 0 for value in unsafe):
            return max(0, cold - sum(unsafe))
    except (ValueError, KeyError):
        pass
    return 0


def capacity(proc=Path("/proc"), *, pid=0):
    try:
        fields = {}
        for line in (proc / "meminfo").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("MemTotal", "MemAvailable"):
                number, unit = value.split()
                if unit != "kB":
                    raise ValueError("unknown memory units")
                fields[name] = int(number) * 1024
        total, free = fields["MemTotal"], fields["MemAvailable"]
        if not 0 <= free <= total or total <= 0:
            raise ValueError("unknown system memory")
        reserve = max(1536 * MIB, min(4 * GIB, total // 5))
        room = free - reserve
        cpus = len(os.sched_getaffinity(pid))
        for group in _groups(proc, pid):
            limits = (_value(group / name) for name in ("memory.max", "memory.high"))
            bounded = [int(limit) for limit in limits if limit not in (None, "max")]
            if bounded:
                limit = min(bounded)
                used = int((group / "memory.current").read_text())
                if used < 0 or limit < 0:
                    raise ValueError("invalid cgroup memory")
                reserve = min(GIB, max(128 * MIB, limit // 8))
                used -= min(used, _inactive_cache(group))
                room = min(room, limit - used - reserve)
            quota = _value(group / "cpu.max")
            if quota is not None:
                amount, period = quota.split()
                if int(period) <= 0 or (amount != "max" and int(amount) <= 0):
                    raise ValueError("invalid cgroup CPU")
                if amount != "max":
                    cpus = min(cpus, max(1, int(amount) // int(period)))
        return Capacity(max(0, room), max(1, cpus))
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return Capacity(0, 1)


def mesh_triangle_budget():
    # Reserve for GTK, the existing scene and renderer setup. A triangle can
    # temporarily occupy both Python objects and several separate GL buffers.
    remaining = max(0, capacity().headroom - 256 * MIB)
    return min(150_000, remaining // 8192)


def model_working_bytes(size, context):
    # Weights plus staging, runner buffers and a context-dependent KV budget.
    return size + size // 4 + 512 * MIB + context * 160 * 1024


def process_tree_rss(pid, proc=Path("/proc")):
    """Count resident pages of a private service and the runners its threads started."""
    seen, pending = set(), {pid}
    pages = 0
    while pending:
        child = pending.pop()
        seen.add(child)
        directory = proc / str(child)
        try:
            pages += int((directory / "statm").read_text().split()[1])
            for thread in (directory / "task").iterdir():
                try:
                    children = (thread / "children").read_text().split()
                except FileNotFoundError:
                    continue
                pending.update(set(map(int, children)) - seen)
        except FileNotFoundError:
            continue  # Processes can finish between samples.
    return pages * os.sysconf("SC_PAGE_SIZE")


def docker_limits():
    cap = capacity()
    # Round to stable 512 MiB steps so common changes in RAM usage do not
    # require recreating an idle container. Even the smallest cap can serve /api/tags.
    step = 512 * MIB
    return min(6 * GIB, max(step, cap.headroom // step * step)), min(4, cap.threads)


if __name__ == "__main__":
    print(*docker_limits())
