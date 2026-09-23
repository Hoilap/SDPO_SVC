import os
import importlib.util
import mmap
import multiprocessing
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The scorer has no dependency on Ray/Torch; load it without initializing verl.
spec = importlib.util.spec_from_file_location(
    "_feedback_code_under_test",
    Path(__file__).resolve().parents[3] / "verl/utils/reward_score/feedback/code.py",
)
code = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = code
spec.loader.exec_module(code)


class _FakeReceiveConnection:
    def __init__(self):
        self.closed = False

    def poll(self, timeout):
        return False

    def close(self):
        self.closed = True


class _FakeSendConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProcess:
    active = 0
    max_active = 0
    instances = []

    def __init__(self, target, args):
        self.alive = False
        self.closed = False
        self.pid = None
        self.__class__.instances.append(self)

    def start(self):
        self.pid = len(self.__class__.instances)
        self.alive = True
        self.__class__.active += 1
        self.__class__.max_active = max(self.__class__.max_active, self.__class__.active)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self.alive

    def kill(self):
        if self.alive:
            self.alive = False
            self.__class__.active -= 1

    def close(self):
        assert not self.alive
        self.closed = True


@pytest.mark.parametrize("concurrency", [3, 8])
def test_run_tests_bounds_concurrency_and_closes_resources(concurrency):
    receive_connections = []
    send_connections = []

    def fake_pipe(duplex):
        assert duplex is False
        receive = _FakeReceiveConnection()
        send = _FakeSendConnection()
        receive_connections.append(receive)
        send_connections.append(send)
        return receive, send

    _FakeProcess.active = 0
    _FakeProcess.max_active = 0
    _FakeProcess.instances = []
    test_cases = {
        "testtype": "functional",
        "fn_name": "answer",
        "inputs": [{} for _ in range(10)],
        "outputs": ["0" for _ in range(10)],
        "time_limit": 0,
    }
    with (
        patch.dict(os.environ, {code.MAX_CONCURRENCY_ENV_VAR: str(concurrency)}),
        patch.object(code.multiprocessing, "Pipe", fake_pipe),
        patch.object(code.multiprocessing, "Process", _FakeProcess),
    ):
        records = code.run_tests(test_cases, "```python\ndef answer():\n    return 0\n```", False, None)

    assert [record["test_idx"] for record in records] == list(range(10))
    assert all(record["actual"] == code.TIMEOUT for record in records)
    assert _FakeProcess.max_active == concurrency
    assert _FakeProcess.active == 0
    assert all(process.closed for process in _FakeProcess.instances)
    assert all(connection.closed for connection in receive_connections)
    assert all(connection.closed for connection in send_connections)


def test_invalid_concurrency_uses_safe_default():
    with patch.dict(os.environ, {code.MAX_CONCURRENCY_ENV_VAR: "0"}):
        assert code._get_max_concurrent_test_processes() == code.DEFAULT_MAX_CONCURRENT_TEST_PROCESSES


def test_send_memory_error_does_not_trigger_recursive_fallback():
    class FailingConnection:
        def __init__(self):
            self.closed = False

        def send(self, record):
            raise MemoryError

        def close(self):
            self.closed = True

    connection = FailingConnection()
    test_cases = {
        "testtype": "functional",
        "fn_name": "answer",
        "inputs": [{}],
        "outputs": ["0"],
    }

    with patch.object(code, "reliability_guard"):
        code.run_tests_for_one_example(
            test_cases,
            "def answer():\n    return 0",
            connection,
            sparse_rewards=False,
            test_idx=0,
        )

    assert connection.closed


def _cases(text="", expected="0", time_limit=2):
    return {"testtype": "stdin", "fn_name": "", "inputs": [text], "outputs": [expected], "time_limit": time_limit}


@pytest.fixture
def fork_scorer(monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux address-space regression")
    monkeypatch.setattr(code, "multiprocessing", multiprocessing.get_context("fork"))
    monkeypatch.setenv("CODE_REWARD_MAX_CONCURRENCY", "8")
    return code


def test_memory_budget_preserves_inherited_limits(monkeypatch):
    baseline = 2 * 1024**3
    budget = 1024**3
    monkeypatch.setattr(code, "_current_vms_bytes", lambda: baseline)
    with patch.object(code._resource, "getrlimit", return_value=(-1, -1)), patch.object(code._resource, "setrlimit") as setter:
        code.set_memory_limits(budget)
        setter.assert_called_once_with(code._resource.RLIMIT_AS, (baseline + budget, baseline + budget))
    for limits in [(baseline + budget - 1, -1), (-1, baseline + budget - 1)]:
        with patch.object(code._resource, "getrlimit", return_value=limits), patch.object(code._resource, "setrlimit") as setter:
            with pytest.raises(code.MemoryLimitSetupError):
                code.set_memory_limits(budget)
            setter.assert_not_called()


def test_memory_setup_failure_is_explicit(monkeypatch):
    def fail():
        raise OSError("unreadable proc")
    monkeypatch.setattr(code, "_current_vms_bytes", fail)
    with pytest.raises(code.MemoryLimitSetupError):
        code.set_memory_limits(1024)


def test_inherited_large_vms_and_large_input(fork_scorer):
    # Reserve virtual space, without physically allocating two GiB.
    with mmap.mmap(-1, 2 * 1024**3):
        result = fork_scorer.run_tests(_cases("0 " * (4 * 1024**2)), "```python\nprint(0)\n```", False, None)[0]
    assert result["passed"]
    assert result["error_kind"] == ""
    assert len(result["input"]) == 8 * 1024**2


@pytest.mark.parametrize("program", [
    "print('x' * 1025)",
    "import sys\nsys.stderr.write('x' * 1025)",
    "debug_print('x' * 1025)",
    "import sys\nsys.stdout.writelines(['x' * 600, 'y' * 600])",
    "try:\n    print('x' * 1025)\nexcept:\n    print(0)",
])
def test_output_limits_at_write_time(fork_scorer, monkeypatch, program):
    monkeypatch.setenv("CODE_REWARD_MAX_OUTPUT_CHARS", "1024")
    result = fork_scorer.run_tests(_cases(), f"```python\n{program}\n```", False, None)[0]
    assert not result["passed"]
    assert result["error_kind"] == "output_limit"


def test_full_answer_compared_before_preview(fork_scorer):
    expected = "x" * 10000
    result = fork_scorer.run_tests(_cases(expected=expected), "```python\nprint('x' * 10000)\n```", False, None)[0]
    assert result["passed"]
    assert len(result["actual"]) == code.MAX_FEEDBACK_CHARS


def test_child_does_not_echo_input_or_expected():
    class Connection:
        def send(self, record):
            self.record = record
        def close(self):
            pass
    conn = Connection()
    with patch.object(code, "reliability_guard"):
        code.run_tests_for_one_example(_cases("x" * 1000000), "print(0)", conn, False, 0)
    assert conn.record["passed"]
    assert "input" not in conn.record and "expected" not in conn.record
    assert len(pickle.dumps(conn.record)) < 1024


def test_send_failure_uses_small_prebuilt_marker():
    class Connection:
        def send(self, record):
            raise MemoryError
        def send_bytes(self, payload):
            self.payload = payload
    conn = Connection()
    code._send_result(conn, {})
    assert len(conn.payload) < 512
    assert pickle.loads(conn.payload)["error_kind"] == "ipc_memory"


def test_execution_memory_is_not_ipc_failure(fork_scorer, monkeypatch):
    monkeypatch.setattr(code, "MAX_ADDITIONAL_MEMORY_BYTES", 32 * 1024**2)
    result = fork_scorer.run_tests(_cases(), "```python\nprint('x' * (256 * 1024**2))\n```", False, None)[0]
    assert not result["passed"]
    assert result["error_kind"] == "execution_memory"


def test_setup_error_is_not_wrong_answer(fork_scorer, monkeypatch):
    def fail():
        raise code.MemoryLimitSetupError("test")
    monkeypatch.setattr(code, "reliability_guard", fail)
    result = fork_scorer.run_tests(_cases(), "```python\nprint(0)\n```", False, None)[0]
    assert result["error_kind"] == "memory_limit_setup"
    assert "Evaluation Infrastructure Error" in code.format_test_feedback([result])


def test_timeout_cleans_up_children(fork_scorer):
    before = {p.pid for p in multiprocessing.active_children()}
    result = fork_scorer.run_tests(_cases(time_limit=0), "```python\nwhile True: pass\n```", False, None)[0]
    assert result["error_kind"] == "timeout"
    assert result["exit_code"] is not None
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_unexpected_exit_is_infrastructure_error(fork_scorer, monkeypatch):
    def exit_child(*args):
        os._exit(23)
    monkeypatch.setattr(code, "run_tests_for_one_example", exit_child)
    result = fork_scorer.run_tests(_cases(), "```python\nprint(0)\n```", False, None)[0]
    assert result["error_kind"] == "process_exit"
    assert result["exit_code"] == 23
    assert "Wrong Answer" not in code.format_test_feedback([result])


def test_functional_and_code_modes(fork_scorer):
    functional = {"testtype": "functional", "fn_name": "answer", "inputs": [{"x": 2}], "outputs": ["3"], "time_limit": 2}
    result = fork_scorer.run_tests(functional, "```python\ndef answer(x): return x + 1\n```", False, None)[0]
    assert result["passed"]
    assert result["input"] == {"x": 2}
    tests = {"testtype": "code", "fn_name": "", "inputs": ["assert answer() == 3"], "outputs": [""], "time_limit": 2}
    result = fork_scorer.run_tests(tests, "```python\ndef answer(): return 3\n```", False, None)[0]
    assert result["passed"]
