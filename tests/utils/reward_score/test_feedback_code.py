import os
from unittest.mock import patch

from verl.utils.reward_score.feedback import code


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


def test_run_tests_bounds_concurrency_and_closes_resources():
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
        patch.dict(os.environ, {code.MAX_CONCURRENCY_ENV_VAR: "3"}),
        patch.object(code.multiprocessing, "Pipe", fake_pipe),
        patch.object(code.multiprocessing, "Process", _FakeProcess),
    ):
        records = code.run_tests(test_cases, "```python\ndef answer():\n    return 0\n```", False, None)

    assert [record["test_idx"] for record in records] == list(range(10))
    assert all(record["actual"] == code.TIMEOUT for record in records)
    assert _FakeProcess.max_active == 3
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
