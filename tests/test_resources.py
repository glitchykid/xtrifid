import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xtrifid import resources


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.proc = root / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.cgroup = root / "groups"
        (self.cgroup / "desktop/task").mkdir(parents=True)
        (self.proc / "meminfo").write_text(
            "MemTotal: 16777216 kB\nMemAvailable: 12582912 kB\nSwapFree: 33554432 kB\n"
        )
        (self.proc / "self/cgroup").write_text("0::/desktop/task\n")
        (self.proc / "self/mountinfo").write_text(
            f"25 19 0:26 / {self.cgroup} ro - cgroup2 cgroup rw\n"
        )
        self.enterContext(patch("os.sched_getaffinity", return_value=set(range(12))))

    def test_parent_and_child_limits_and_cpu_affinity(self):
        group = self.cgroup / "desktop"
        leaf = group / "task"
        (group / "memory.max").write_text(str(4 << 30))
        (group / "memory.high").write_text(str(3 << 30))
        (group / "memory.current").write_text(str(2 << 30))
        (group / "cpu.max").write_text("250000 100000\n")
        (leaf / "memory.max").write_text("max")
        (leaf / "cpu.max").write_text("max 100000")
        cap = resources.capacity(self.proc)
        self.assertEqual(cap.cpus, 2)
        self.assertGreater(cap.headroom, 0)
        self.assertLess(cap.headroom, 1 << 30)
        (group / "memory.current").write_text(str(3 << 30))
        self.assertEqual(resources.capacity(self.proc).headroom, 0)

    def test_unbounded_v2_reserves_host_memory_and_ignores_swap(self):
        cap = resources.capacity(self.proc)
        self.assertEqual(cap.cpus, 12)
        self.assertGreater(cap.headroom, 8 << 30)
        self.assertLess(cap.headroom, 11 << 30)
        (self.proc / "meminfo").write_text(
            "MemTotal: 16777216 kB\nMemAvailable: 200000 kB\nSwapFree: 33554432 kB\n"
        )
        self.assertEqual(resources.capacity(self.proc).headroom, 0)

    def test_unknown_or_unreadable_resources_fail_closed(self):
        for path, value in (
            (self.cgroup / "desktop/memory.max", "1234567"),
            (self.cgroup / "desktop/memory.max", "unknown"),
            (self.proc / "self/cgroup", "0::/missing"),
        ):
            with self.subTest(path=path, value=value):
                path.write_text(value)
                self.assertEqual(resources.capacity(self.proc).headroom, 0)
                path.unlink()
        (self.proc / "meminfo").unlink()
        self.assertEqual(resources.capacity(self.proc).headroom, 0)

    def test_mesh_budget_declines_with_capacity_and_has_a_hard_ceiling(self):
        def budget(free):
            with patch.object(resources, "capacity", return_value=resources.Capacity(free, 2)):
                return resources.mesh_triangle_budget()

        self.assertEqual(budget(0), 0)
        self.assertGreater(budget(512 << 20), 0)
        self.assertLess(budget(512 << 20), budget(2 << 30))
        self.assertLessEqual(budget(1 << 50), 150000)

    def test_runner_memory_includes_workers_started_by_any_thread(self):
        parent = self.proc / "21"
        child = self.proc / "25"
        for process, thread, children, pages in (
            (parent, "21", "", 100),
            (parent, "22", "25 27", 100),
            (child, "25", "", 250),
        ):
            job = process / "task" / thread
            job.mkdir(parents=True)
            (job / "children").write_text(children)
            (process / "statm").write_text(f"9999 {pages} 1 2 3 4 5")
        self.assertEqual(
            resources.process_tree_rss(21, self.proc), 350 * os.sysconf("SC_PAGE_SIZE")
        )

    def test_sibling_runner_uses_its_quota_and_only_clean_unmapped_inactive_cache(self):
        (self.proc / "99").mkdir()
        (self.proc / "99/cgroup").write_text("0::/docker\n")
        group = self.cgroup / "docker"
        group.mkdir()
        (group / "memory.max").write_text(str(4 << 30))
        (group / "cpu.max").write_text("200000 100000")
        (group / "memory.current").write_text(str(3 << 30))
        stats = {
            "inactive_file": 2 << 30,
            "file_mapped": 128 << 20,
            "file_dirty": 64 << 20,
            "file_writeback": 32 << 20,
            "shmem": 32 << 20,
        }
        (group / "memory.stat").write_text("".join(f"{k} {v}\n" for k, v in stats.items()))
        cap = resources.capacity(self.proc, pid=99)
        self.assertEqual(cap.cpus, 2)
        self.assertEqual(cap.headroom, (2 << 30) + (256 << 20))
        (group / "memory.stat").write_text("inactive_file 2147483648\n")
        self.assertEqual(resources.capacity(self.proc, pid=99).headroom, 512 << 20)

    def test_docker_hard_limits_scale_down_to_the_current_machine(self):
        for free, cpus, expected in (
            (100 << 20, 1, (512 << 20, 1)),
            (3 << 30, 4, (3 << 30, 2)),
            (20 << 30, 16, (6 << 30, 4)),
        ):
            with patch.object(
                resources, "capacity", return_value=resources.Capacity(free, cpus)
            ):
                self.assertEqual(resources.docker_limits(), expected)
