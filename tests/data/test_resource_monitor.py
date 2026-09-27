"""Cgroup attribution checks for the independent resource sampler."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "resource_monitor", ROOT / "experiments/continual/monitor_resources.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class ResourceMonitorTest(unittest.TestCase):
    def test_root_controller_does_not_include_other_jobs(self):
        scopes = monitor.select_scopes(
            {"name=systemd": "/", "memory": "/slurm/uid_1000/job_123/step_batch"}, "123"
        )
        self.assertEqual(scopes, {"memory": "/slurm/uid_1000/job_123"})
        scope = scopes["memory"]
        self.assertTrue(monitor.within("/slurm/uid_1000/job_123/step_0", scope))
        self.assertFalse(monitor.within("/slurm/uid_1000/job_1234/step_0", scope))

    def test_unified_cgroup_scope(self):
        own = monitor.memberships("0::/slurm/job_123.scope/step_batch\n")
        self.assertEqual(monitor.select_scopes(own, "123"), {"": "/slurm/job_123.scope"})

    def test_outside_slurm_keeps_current_membership(self):
        own = {"": "/user.slice/session.scope"}
        self.assertEqual(monitor.select_scopes(own, ""), own)


if __name__ == "__main__":
    unittest.main()
