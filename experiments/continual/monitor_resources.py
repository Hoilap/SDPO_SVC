"""Independent, node-local Slurm worker/cgroup sampler (no Ray/W&B dependency)."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def memberships(text):
    result = {}
    for line in (text or "").splitlines():
        _, controllers, path = line.split(":", 2)
        result[controllers] = path
    return result


def job_scope(path, job_id):
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in (f"job_{job_id}", f"job_{job_id}.scope"):
            return str(Path(*parts[:index + 1]))
    return path


def within(path, scope):
    return path == scope or path.startswith(scope.rstrip("/") + "/")


def select_scopes(own, job_id):
    # Generic controllers (e.g. name=systemd at /) must not match other jobs
    # when a Slurm job boundary is available in cpuset/memory/freezer.
    job_scopes = {}
    for key, path in own.items():
        scope = job_scope(path, job_id)
        if job_id and Path(scope).name in (f"job_{job_id}", f"job_{job_id}.scope"):
            job_scopes[key] = scope
    return job_scopes or own


def sample(scopes, mounts):
    record = {"timestamp": time.time(), "monotonic": time.monotonic(), "workers": [], "cgroups": {}}
    record["node"] = {name: read("/proc/" + name) for name in
                      ("meminfo", "vmstat", "loadavg", "stat", "diskstats",
                       "pressure/cpu", "pressure/memory", "pressure/io")}
    groups = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            member = memberships(read(proc / "cgroup"))
            if not any(key in member and within(member[key], scope) for key, scope in scopes.items()):
                continue
            # Deliberately omit environment and command line (may contain credentials).
            record["workers"].append({"pid": int(proc.name), **{
                name: read(proc / name) for name in
                ("comm", "status", "stat", "statm", "smaps_rollup", "io", "wchan", "cgroup")}})
            for controllers, path in member.items():
                for root, mount, options, unified in mounts:
                    if (unified and controllers == "") or (not unified and set(controllers.split(",")) & options):
                        if not within(path, root):
                            continue
                        resolved = Path(mount) / os.path.relpath(path, root)
                        # Include parent limits: the job-level cap may be above the step.
                        while within(str(resolved), mount):
                            groups.add(resolved)
                            if str(resolved) == mount:
                                break
                            resolved = resolved.parent
        except (OSError, ValueError):
            continue  # Processes can exit during a sample.
    counters = ("memory.current", "memory.max", "memory.high", "memory.peak", "memory.events",
                "memory.events.local", "memory.stat", "memory.swap.current", "memory.swap.max",
                "memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes",
                "memory.failcnt", "memory.oom_control", "memory.memsw.usage_in_bytes",
                "memory.memsw.limit_in_bytes", "cpu.stat", "cpu.max", "cpuacct.usage",
                "cpu.cfs_quota_us", "cpu.cfs_period_us", "cpuset.cpus", "cpuset.mems",
                "cpuset.cpus.effective", "cpuset.mems.effective", "pids.current", "pids.max",
                "io.stat", "blkio.throttle.io_service_bytes", "cgroup.procs")
    for path in sorted(groups):
        record["cgroups"][str(path)] = {key: value for key in counters
                                      if (value := read(path / key)) is not None}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    own = memberships(read("/proc/self/cgroup"))
    scopes = select_scopes(own, os.environ.get("SLURM_JOB_ID", ""))
    mounts = []
    for line in (read("/proc/self/mountinfo") or "").splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] in ("cgroup", "cgroup2"):
            mounts.append((fields[3], fields[4], set(fs[2].split(",")), fs[0] == "cgroup2"))
    stop = False

    def finish(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a", buffering=1) as stream:
        stream.write(json.dumps({"event": "monitor_start", "interval": args.interval,
                                 "job_id": os.environ.get("SLURM_JOB_ID"), "scopes": scopes,
                                 "mounts": [(r, m, sorted(o), u) for r, m, o, u in mounts]}) + "\n")
        deadline = time.monotonic()
        while not stop:
            start = time.monotonic()
            record = sample(scopes, mounts)
            record["sampling_seconds"] = time.monotonic() - start
            stream.write(json.dumps(record) + "\n")
            if args.once:
                break
            deadline += args.interval
            while not stop and time.monotonic() < deadline:
                time.sleep(max(0, min(0.2, deadline - time.monotonic())))
            if time.monotonic() > deadline + args.interval:
                deadline = time.monotonic()


if __name__ == "__main__":
    main()
